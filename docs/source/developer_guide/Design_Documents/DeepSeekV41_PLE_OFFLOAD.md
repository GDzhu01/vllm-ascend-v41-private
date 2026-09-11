# DeepSeek V4.1 Engram PLE_OFFLOAD

PLE_OFFLOAD 将压缩的 Engram 表保留在 CPU：CPU 执行 hash、查表和 group32 反量化，命中行以 BF16 传回 NPU，`wkv` 和 gate 仍在 NPU 执行。节点内行路由沿用 HCCL，不把完整 Engram 表展开到 HBM，也不改变 gate 的旋转和数值接口。

对于 `DeepSeek-V4.1-Flash-W8A8` 中的 INT8 Engram，通过 `--additional-config` 显式开启：

```json
{"enable_engram_ple_offload": true, "engram_storage": "int8"}
```

INT8 weight 为 `[rows,256]`，FP32 scale 为 `[rows,8]`，两者都按节点内 rank 切分并驻留普通 CPU 内存；仅命中的 BF16 行使用带事件保护的双槽 pinned 缓冲。FP32 反量化后直接写入 BF16 缓冲，CPU 路径不调用 NPU Triton gather kernel。未开启 PLE 的 INT8 存储仍在 NPU。

原 FP8 模式继续支持。仅开启 PLE 而不指定存储时仍默认 FP8；当主模型目录中的 Engram 是 BF16、原始 FP8 表位于另一目录时，指定表目录：

```json
{
  "enable_engram_ple_offload": true,
  "engram_model_path": "/path/to/original-fp8-checkpoint"
}
```

该路径只用于读取 Engram weight 和 `.scale`；主模型的 tokenizer、QuaRot 和其他权重仍从模型目录读取。

loader 读取 `model.safetensors.index.json` 或 `quant_model_weights.safetensors.index.json`。INT8 源表要求 I8 weight 与 FP32 group32 scale；FP8 源表要求 `F8_E4M3` weight 与 `F8_E8M0`/`F8_E8M0FNU` scale。INT8 存储也保留从 BF16 源表逐块量化的行为，此时不要求 checkpoint 提供 scale。模型主干仍按原有量化配置加载；Engram 查表输出固定为 BF16。

运行时需要 `--safetensors-load-strategy lazy`，以及现有 TP8、EP、PP=PCP=DCP=1 和节点内查询组约束。A3 单机 16 卡对应 TP8/DP2/EP16，需暴露全部 16 卡。

验证覆盖配置、CPU 分片、INT8/FP8 loader、BF16 位模式、变 batch/idle DP、pinned 缓冲复用与事件等待。组件通过不等于 TP8/DP2 整模型验收；主模型 FULL_DECODE_ONLY、DSpark eager、HTTP、精度和吞吐需在目标服务上单独验证。
