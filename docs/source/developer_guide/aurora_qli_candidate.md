# Aurora QLI V2 and candidate integration

Aurora's indexer uses `QuantLightningIndexerV2` with paged INT8 index K.
The imported operator source is from `ops-transformer-qli_candidate.zip`,
SHA256 `771f0c16b9119c676c10cebef713b168f127ff194f3f2f98edcbcd0979c6f966`.
Its companion `QuantLightningIndexerV2Metadata` is built and registered too.

## Data flow

`DeepseekV41Indexer.select` projects Q and head weights and applies RoPE.
`select_projected` quantizes each Q head to INT8 with a representable FP16
scale. Head weights and K scales are FP16. Existing source-owned K-cache
updates remain unchanged.

The operator consumes `TND` Q and `PA_BBND` index K directly. Aurora's
four-slot layer-outermost allocator packs index K and FP16 scales after KV
inside each shared page. C2 uses 64-row views with a 131072-byte block stride;
C1 uses 128-row views with a 147712-byte block stride. The Torch adapter passes
their actual leading strides to ACLNN.
Neither layout requires gathering the whole context into a dense tensor.

Metadata receives original query boundaries, compressed K lengths, and the
original sequence length modulo the compression ratio. The residual tensor
must be absent for ratio 1. BSND callers must omit `cu_seqlens_q`.

The three candidate modes share one native operator:

| Mode | Meaning | Result |
| --- | --- | --- |
| 1 | Candidate source | Unfiltered position TopK and candidate block IDs |
| 2 | Candidate consumer | Rerank using this layer's Q and weights within source blocks |
| 3 | Candidate disabled | Ordinary position TopK |

Candidates are INT32 block IDs of shape `[tokens, 1, candidate_topk_blocks]`,
with `-1` padding. They are neither index-K vectors nor position TopK. A block
contains 8 compressed positions. Block scores use the maximum position score;
the last visible block is pinned. Shared attention state retains this tensor
within one forward and resets it on the next forward.

The model sorts returned position indices chronologically and moves `-1`
padding to the end before attention. Empty compressed contexts return empty
indices and, for a source, all-invalid candidates. A consumer without a source
raises an error.

## Current A3 contract

- INT8 Q/K, FP16 head weights and per-head Q/per-token K scales; quant mode 2.
- 32 or 64 replicated index heads, head dimension 128, one index-K head.
- Aurora compression ratios 1 and 2, causal mask mode 3.
- Position TopK in `[1, 2048]`; candidate blocks a multiple of 64 in `[64, 2048]`.
- Candidate block size is exactly 8 in this kernel implementation.
- TND and BSND operator layouts; model integration uses TND.
- The A3 operator returns indices and candidate IDs, not score values.

The numerical reference follows the supplied INT8 golden: INT32 QK divided by
1024, FP16 ReLU and `weight * Q_scale`, FP32 head reduction, then K scaling.
Query quantization and FP16 weight rounding differ from the earlier floating-Q
small-operator path. Operator agreement with this quantized reference does not
establish full-model or dataset accuracy.

## Build and regression coverage

Build with `pip install -v -e . --no-deps --no-build-isolation` on the paired
CANN/NPU environment. Both new symbols are registered on PrivateUse1 and Meta.

`tests/e2e/nightly/single_node/ops/singlecard_ops/test_deepseek_v41_qli.py`
checks candidate generation/consumption, different consumer queries, ratio
boundaries, paged views, mixed requests, 2048 candidate blocks, 64 heads, empty
contexts and Meta shapes against an independent CPU reference. Ties at the
TopK cutoff use score validity, uniqueness and count rather than arbitrary
index ordering.

The mixed-request model-indexer cases allocate the actual four-slot cache
configuration, including the null ID, and test source/consumer selection on
its strided index K/scale views. The SparseFlashMla suite also compares the
slot-backed BF16 views against the earlier block-outermost layout for ratios
0/1/2 and decode, prefill and mixed requests. These updated tests have not yet
been executed; remote torch/NPU verification is deferred. The end-to-end
results below describe the original layout.

The imported host tiling needed one semantic fix: TND candidate size is
`T * N_k * blocks`, because T already includes every request. Multiplying by
batch size again incorrectly rejects mixed-request consumers. Kernel offsets
already use TND query prefixes and require no corresponding change.

Validation includes single-chip eager and Meta contracts, plus the full
40-layer W8A8 checkpoint on one A3 server with TP4/DP4/EP16 (Engram and
DSpark disabled). Four end-to-end cases were each run twice at temperature 0,
seed 7: arithmetic, capital-city lookup, multi-turn recall, and a 20,032-token
retrieval prompt. Both runs returned the expected answers and identical output
tokens. The long prompt exceeds the 16,384-position candidate block budget.
It exercises chunked prefill and decode with actual candidate filtering.

The checkpoint-provided `encoding.encode_messages(..., thinking_mode="chat")`
was used with `/v1/completions`; the checkpoint does not provide a standard
chat template. Repeated output agreement validates these fixed functional
cases, not a baseline-versus-candidate dataset accuracy comparison. Graph,
64K/128K requests, multi-node execution, Engram and DSpark remain unvalidated.
