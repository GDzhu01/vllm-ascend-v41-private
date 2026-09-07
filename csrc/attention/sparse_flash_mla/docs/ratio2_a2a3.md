# A2/A3 cmp_ratio=2 适配说明

## 1. 范围与结果

本次仅为前向 SparseFlashMla 的 CSA 增加压缩倍率 2，配套修改 SparseFlashMlaMetadata。A2/A3 HCA 保持仅支持 128 的原有逻辑。950 的倍率范围不变；不涉及梯度算子、压缩算子或上游索引器的实现。

| A2/A3 模式 | 修改前 | 修改后 |
| --- | --- | --- |
| SWA，无 cmp KV | 1 | 1 |
| CSA，有 cmp KV 和 cmp 索引 | 4 | 2、4 |
| HCA，有 cmp KV、无 cmp 索引 | 128 | 128 |

其余约束保持现有实现，例如 cmp causal mask 为 3、CSA TopK 容量为 512/1024、ori 窗口为左 127/右 0。ratio=2 不意味着扩大其他输入规格。

## 2. 倍率在代码中的传递

```text
调用方 cmp_ratio=2
  Metadata Host: IsCmpRatioSupportSmla → ParamsCheck
    AICPU: cmpRatio_ → GetRevertS2Size → CalcCmpBlockRange
      → block/cost → 分核 → metadata
  主算子 Host: CheckSingleParaCmpRatio → cmpParams.cmpRatio
    arch22 Kernel: constInfo.cmpRatio
      → 压缩有效范围 → gather / 逐行 mask → attention
```

倍率是运行时 tiling 参数，不是 tiling key 的模板维度。两次调用必须使用同样的倍率、有效长度、residual，改变这些参数后应重新生成 metadata。

## 3. 适配点与实现决策

| 层次 | 文件 / 符号 | 本次处理 |
| --- | --- | --- |
| 主算子 Host | [sparse_flash_mla_tiling.cpp](../op_host/sparse_flash_mla_tiling.cpp)，`CheckSingleParaCmpRatio` | 仅 CSA 增加 2；保持 HCA=128、SWA=1，更新 CSA 报错 |
| Metadata Host | [metadata_check.h](../../sparse_flash_mla_metadata/op_host/sparse_flash_mla_metadata_check.h)，`IsCmpRatioSupportSmla` | 与主算子允许集合一致，更新错误信息 |
| Metadata AICPU | [metadata_aicpu.cpp](../../sparse_flash_mla_metadata/op_kernel_aicpu/sparse_flash_mla_metadata_aicpu.cpp) | 保留已有运行时乘除公式、residual 范围校验及 block/cost 逻辑 |
| CSA Kernel | [csa_kernel.h](../op_kernel/arch22/sparse_flash_mla_csa_kernel.h) | 已通过 cmpRatio 计算长度和 cmpS2IdLimit，无需新增 ratio 模板 |
| CSA gather | [csa_block_vector.h](../op_kernel/arch22/sparse_flash_mla_csa_block_vector.h) | 沿用 cmpS2IdLimit 检查索引，重点验证 causal 边界 |
| HCA Kernel / mask | [swa_kernel.h](../op_kernel/arch22/sparse_flash_mla_swa_kernel.h) | 保持原有实现，不增加倍率 2 支持 |
| tiling / 内存 | Host SplitBalanced、DoOpTiling | 保持 S2=512 和原缓冲区配置；增加序列长度会增加循环次数，不直接扩大片上基本块 |
| 接口 / 绑定 | 现有整数属性透传 | 不修改 ABI、输出 shape、dtype 或 tiling key |

Kernel 与 AICPU 的相关公式已经参数化，所以本次不为“有 Kernel 变更”而改写等价计算。真正的生产代码变更是两处 Host 倍率白名单。

## 4. 长度、residual 与 causal 边界

令 Lc 为压缩后有效长度，r 为倍率，residual 为余数：

```text
L = Lc * r + residual
p = L - Lq + query_index
visible_cmp = clamp((p + 1) / r, 0, Lc)
```

非负坐标下除法向下取整，负范围由具体分支裁剪。r=2 时 residual 只能为 0 或 1。cmp_mask_mode=3 且 r!=1 时 residual 必须同时传给 Metadata 和主算子，即使余数为 0。

例：原始各 batch 长度 [3,3]，压缩有效长度为 [1,1]，residual 为 [1,1]，压缩 TND 前缀和为 [0,1,2]。不能将 ori 前缀和 [0,3,6] 逐项除以 2 得到 [0,1,3] 后当作压缩前缀和。

CSA 调用方必须实际生成倍率 2 的 KV、索引和分页表。相同原始长度下，从倍率 4 改为 2 会增加压缩 KV 条目数量，应重新预算输入 cache；不能只改变算子属性。A2/A3 的 HCA 仍传 128。

## 5. 测试与验收

新增/扩展的测试：

- [Host tiling UT](../tests/ut/op_host/arch35/test_sparse_flash_mla_tiling.cpp)：使用 Ascend910B、Ascend910_93 case，覆盖 CSA=2 成功、HCA=2 拒绝、旧倍率、非法倍率、缺少 residual 和 SWA=2 拒绝。文件位于 arch35 子目录，但其已有测试框架可构造 A2/A3 平台上下文，并由父 CMake 纳入 Host UT。
- [Metadata API UT](../../sparse_flash_mla_metadata/tests/ut/op_host/op_api/test_aclnn_sparse_flash_mla_metadata.cpp)：通过 ParamsCheck 覆盖两种平台与压缩场景，检查主接口/前置接口规则一致。
- [ratio2 数值回归](../tests/pytest/test_sparse_flash_mla_ratio2.py)：48 个 CSA 组合，覆盖 FP16/BF16、BSND/TND/PA_BBND、residual=0/1、压缩长度 1/511/512/513、双 batch 和多 query 行，同时比较 attn_out 与 LSE。分页采用 ori=128/cmp=16 的不同页大小，CSA 使用 1024 容量以覆盖跨 S2 块。

在已配置本仓库依赖的 Linux/Ascend 环境执行，A2 和 A3 均需验证：

```bash
bash build.sh --ophost_test --ops=sparse_flash_mla --soc=ascend910b --incremental
bash build.sh --opapi_test --ops=sparse_flash_mla_metadata --soc=ascend910b --incremental
cd attention/sparse_flash_mla/tests/pytest
pytest -q test_sparse_flash_mla_ratio2.py
```

上板前须重新编译/安装修改后的主算子与 Metadata。A3 使用构建系统对应的 ascend910_93 目标；已有 ratio=4/128 数值用例也需回归。Host UT 不执行 AICPU 任务切分，新增数值回归通过真实 Metadata + 主算子调用覆盖该部分。

本次环境为 Windows，未发现可用 Python、CMake、Bash 或 WSL Linux 发行版，不能编译 CANN UT 或执行 NPU 数值测试。已执行修改内容、链接及差异的静态检查；上板正确性和性能尚待上述测试确认。

## 6. 已知边界与后续验证

arch22 的 TND 压缩长度读取当前直接使用 cu_seqlens_cmp_kv 相邻差值，Metadata 则优先使用 seqused_cmp_kv。对“有效长度小于存储长度”的 TND 输入，这两个口径需要另行统一。本次不改变原有长度接口语义，新增 TND 用例使用二者一致的长度，不将这种带 padding 的有效长度覆盖场景声明为已解决。

CSA 多核调度、G=1/128、非均匀 batch、aclgraph、极端空范围还应在目标平台验收时扩展覆盖。普通模式本次保持现有流水和内存设计，性能结论必须由 profiling 给出。
