// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <functional>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr int64_t kGroupSize = 32;

// Host lookup keeps its own thread budget so the Engram budget never resizes
// the process-wide intra-op pool used by every other CPU operator.
int64_t lookup_thread_count() {
    static const int64_t value = [] {
        const char* text = std::getenv("ENGRAM_CPU_LOOKUP_THREADS");
        if (text == nullptr || *text == '\0') {
            return int64_t{0};
        }
        char* end = nullptr;
        const long parsed = std::strtol(text, &end, 10);
        if (end == text || parsed <= 0) {
            return int64_t{0};
        }
        return static_cast<int64_t>(parsed);
    }();
    return value;
}

std::vector<int> parse_cpu_list(const std::string& spec) {
    std::vector<int> cpus;
    size_t start = 0;
    while (start < spec.size()) {
        const size_t comma = spec.find(',', start);
        const std::string item = spec.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        const size_t dash = item.find('-');
        if (dash == std::string::npos) {
            cpus.push_back(std::atoi(item.c_str()));
        } else {
            const int first = std::atoi(item.substr(0, dash).c_str());
            const int last = std::atoi(item.substr(dash + 1).c_str());
            for (int cpu = first; cpu <= last; ++cpu) {
                cpus.push_back(cpu);
            }
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return cpus;
}

std::vector<int> lookup_cpu_list() {
    static const std::vector<int> value = [] {
        const char* text = std::getenv("ENGRAM_CPU_LOOKUP_CPUS");
        if (text == nullptr || *text == '\0') {
            return std::vector<int>();
        }
        return parse_cpu_list(text);
    }();
    return value;
}

// Below this row count the synchronous intra-op path beats the pool: a call
// only needs a few chunks, so the worker hand-off dominates.  The crossover
// depends on the process intra-op width: vLLM normally pins it to one thread
// (OMP_NUM_THREADS=1), where the pool already wins from 256 rows up, while a
// process that keeps ~20 intra-op threads wins only above ~4k rows.
int64_t lookup_pool_min_rows() {
    static const int64_t value = [] {
        const char* text = std::getenv("ENGRAM_CPU_LOOKUP_MIN_ROWS");
        if (text != nullptr && *text != '\0') {
            char* end = nullptr;
            const long parsed = std::strtol(text, &end, 10);
            if (end != text && parsed >= 0) {
                return static_cast<int64_t>(parsed);
            }
        }
        return at::get_num_threads() <= 2 ? int64_t{256} : int64_t{4096};
    }();
    return value;
}

// A persistent pool that runs the row loop on a caller-supplied budget.  The
// caller participates, so ``threads`` is the total width.  Submissions from
// several Python workers serialize; the lookup has one batch in flight per
// layer and concurrent layers are rare.
class LookupPool {
public:
    static LookupPool& get() {
        static LookupPool pool;
        return pool;
    }

    bool enabled() const { return threads_ > 0; }

    void parallel_for(int64_t begin, int64_t end, int64_t grain, const std::function<void(int64_t, int64_t)>& body) {
        if (!enabled() || end <= begin || grain <= 0) {
            if (end > begin) {
                body(begin, end);
            }
            return;
        }
        std::lock_guard<std::mutex> job_lock(job_mutex_);
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            body_ = &body;
            end_ = end;
            grain_ = grain;
            next_.store(begin, std::memory_order_relaxed);
            remaining_ = threads_ + 1;
            error_ = nullptr;
            ++job_;
            work_cv_.notify_all();
        }
        run_ranges();
        finish_range();
        std::unique_lock<std::mutex> lock(state_mutex_);
        done_cv_.wait(lock, [this] { return remaining_ == 0; });
        if (error_ != nullptr) {
            auto error = error_;
            error_ = nullptr;
            lock.unlock();
            std::rethrow_exception(error);
        }
    }

private:
    LookupPool() {
        const int64_t requested = lookup_thread_count();
        if (requested <= 0) {
            return;
        }
        threads_ = static_cast<int>(requested);
        cpu_list_ = lookup_cpu_list();
        workers_.reserve(threads_);
        for (int index = 0; index < threads_; ++index) {
            workers_.emplace_back([this, index] { worker_loop(index); });
        }
    }

    ~LookupPool() {
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            stopping_ = true;
            ++job_;
            work_cv_.notify_all();
        }
        for (auto& worker : workers_) {
            worker.join();
        }
    }

    void worker_loop(int index) {
        if (!cpu_list_.empty()) {
            cpu_set_t mask;
            CPU_ZERO(&mask);
            CPU_SET(cpu_list_[index % cpu_list_.size()], &mask);
            pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
        }
        uint64_t seen = 0;
        while (true) {
            {
                std::unique_lock<std::mutex> lock(state_mutex_);
                work_cv_.wait(lock, [this, seen] { return stopping_ || job_ != seen; });
                if (stopping_) {
                    return;
                }
                seen = job_;
            }
            run_ranges();
            finish_range();
        }
    }

    void run_ranges() {
        while (true) {
            const int64_t start = next_.fetch_add(grain_, std::memory_order_relaxed);
            if (start >= end_) {
                return;
            }
            const int64_t stop = std::min(start + grain_, end_);
            try {
                (*body_)(start, stop);
            } catch (...) {
                std::lock_guard<std::mutex> lock(state_mutex_);
                if (error_ == nullptr) {
                    error_ = std::current_exception();
                }
            }
        }
    }

    void finish_range() {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (--remaining_ == 0) {
            done_cv_.notify_all();
        }
    }

    int threads_ = 0;
    std::vector<int> cpu_list_;
    std::vector<std::thread> workers_;
    std::mutex job_mutex_;
    std::mutex state_mutex_;
    std::condition_variable work_cv_;
    std::condition_variable done_cv_;
    const std::function<void(int64_t, int64_t)>* body_ = nullptr;
    std::atomic<int64_t> next_{0};
    int64_t end_ = 0;
    int64_t grain_ = 0;
    int remaining_ = 0;
    uint64_t job_ = 0;
    bool stopping_ = false;
    std::exception_ptr error_;
};

#if defined(__aarch64__)
uint16x4_t to_bfloat16(float32x4_t values) {
    const auto bits = vreinterpretq_u32_f32(values);
    const auto bias = vaddq_u32(vdupq_n_u32(0x7fff), vandq_u32(vshrq_n_u32(bits, 16), vdupq_n_u32(1)));
    const auto rounded = vshrn_n_u32(vaddq_u32(bits, bias), 16);
    // Match c10::BFloat16's canonical NaN and round-to-nearest-even.
    return vbsl_u16(vmovn_u32(vceqq_f32(values, values)), rounded, vdup_n_u16(0x7fc0));
}
#endif

void engram_int8_lookup_cpu(const at::Tensor& weight, const at::Tensor& scale,
                          const at::Tensor& ids, at::Tensor& output) {
    for (const auto& tensor : {weight, scale, ids, output}) {
        TORCH_CHECK(tensor.device().is_cpu() && tensor.is_contiguous(),
                    "Engram CPU lookup requires contiguous CPU tensors");
    }
    TORCH_CHECK(weight.scalar_type() == at::kChar && scale.scalar_type() == at::kFloat &&
                ids.scalar_type() == at::kLong && output.scalar_type() == at::kBFloat16,
                "Engram CPU lookup expects INT8 weight, FP32 scale, INT64 IDs and BF16 output");
    TORCH_CHECK(weight.dim() == 2 && scale.dim() == 2 && ids.dim() == 1 && output.dim() == 2,
                "Invalid Engram CPU lookup dimensions");
    const auto rows = weight.size(0);
    const auto width = weight.size(1);
    const auto groups = width / kGroupSize;
    TORCH_CHECK(width > 0 && width % kGroupSize == 0 && scale.size(0) == rows &&
                scale.size(1) == groups && output.size(0) == ids.numel() && output.size(1) == width,
                "Invalid Engram CPU lookup shapes");
    const auto* codes = weight.const_data_ptr<int8_t>();
    const auto* scales = scale.const_data_ptr<float>();
    const auto* indices = ids.const_data_ptr<int64_t>();
    auto* out = output.mutable_data_ptr<at::BFloat16>();
    // Tiny decode batches do not amortize the thread-pool handoff.  A grain
    // equal to the whole request keeps this path inline while retaining
    // parallel row processing for prefill-sized lookups.
    const auto grain = ids.numel() < 256 ? ids.numel() : 128;
    const auto run_rows = [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const auto row = indices[i];
            TORCH_CHECK_INDEX(row >= 0 && row < rows, "Engram CPU lookup ID outside shard");
            for (int64_t g = 0; g < groups; ++g) {
                const float factor = scales[row * groups + g];
#if defined(__aarch64__)
                for (int64_t j = 0; j < kGroupSize; j += 8) {
                    const auto column = g * kGroupSize + j;
                    const auto values = vmovl_s8(vld1_s8(codes + row * width + column));
                    const auto low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(values)));
                    const auto high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(values)));
                    auto* destination = reinterpret_cast<uint16_t*>(out + i * width + column);
                    vst1_u16(destination, to_bfloat16(vmulq_n_f32(low, factor)));
                    vst1_u16(destination + 4, to_bfloat16(vmulq_n_f32(high, factor)));
                }
#else
                for (int64_t j = 0; j < kGroupSize; ++j) {
                    const auto column = g * kGroupSize + j;
                    out[i * width + column] = at::BFloat16(float(codes[row * width + column]) * factor);
                }
#endif
            }
        }
    };
    auto& pool = LookupPool::get();
    if (pool.enabled() && ids.numel() >= lookup_pool_min_rows()) {
        pool.parallel_for(0, ids.numel(), grain, run_rows);
    } else {
        at::parallel_for(0, ids.numel(), grain, run_rows);
    }
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_C_ascend, m) {
    m.def("engram_int8_lookup_cpu(Tensor weight, Tensor scale, Tensor ids, Tensor(a!) output) -> ()");
}

TORCH_LIBRARY_IMPL(_C_ascend, CPU, m) {
    m.impl("engram_int8_lookup_cpu", &engram_int8_lookup_cpu);
}
