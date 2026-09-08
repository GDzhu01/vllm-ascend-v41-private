# DeepSeek V4.1 eager bring-up

The V4.1 backbone is registered through `model.py`, matching the package
structure used by DeepSeek V4. There is no separate `modeling.py` and no
model-side KV-cache owner.

## Current execution path

- `model.py` owns the V4.1 backbone, delayed mHC handoff, attention projections,
  source references, compressor invocation and the correctness-first eager
  attention path.
- `attention/dsa_v41.py` owns paged-cache registration and metadata plus the
  unfused SWA/long-KV gather, attention and cache-scatter operations.
- `core/deepseek_v41.py` owns cache specs, hybrid grouping, sizing, allocation
  and reshape.
- `compressor.py` owns ratio1/ratio2 compressor parameters and the ratio2
  FP32 state cache. `indexer.py` owns Index-K cache updates, Indexer scoring,
  candidate-block filtering and chronological TopK selection.

The model reuses DeepSeek V4's quantization-aware projection, MoE and output
projection implementations. Small operators replace the fused DSA kernel for
the initial eager milestone.

## Hybrid cache layout

The allocator follows the DeepSeek V4 layer-tuple pattern with one global
block-ID lifecycle and four separate layer-outermost buffers. Each buffer
contains all blocks for one shared slot; each block contains the owning
group's resource tuple. Different groups own distinct simultaneous live IDs.
Within the merged group, KV and index share an ID at disjoint byte offsets.

| Group | Resources | Logical block size | Slots |
| --- | --- | --- | --- |
| G0 | C2 KV/index at layers 2, 8, 14; C1 KV/index at layer 20 | 128 | 0-3 |
| G1 | FP32 compressor state at layers 2, 8, 14, window 2 | 16 | 0-2; slot 3 unused |
| G2-G11 | SWA layers 0-39, four consecutive layers per group, window 128 | 128 | 0-3 |

There are 12 groups and 51 cache specs. The base attention block size is
128 at production dimensions. C2 stores 64 compressed rows per logical block;
C1 and SWA store 128 rows. State stores 16 uncompressed rows with width 1024.
C2 and C1 share the original-token block table, with compression applied by
per-layer metadata builders. No paired-block mapping is needed.

For MLA width 512, index width 128 and one KV head:

| Slot | Full-context tuple | Page bytes | Component offsets (bytes) |
| --- | --- | --- | --- |
| 0-2 | C2 KV + INT8 index K + FP16 scales | 131072 | KV 0; K 65536; scale 73728 |
| 3 | C1 KV + INT8 index K + FP16 scales | 147712 | KV 0; K 131072; scale 147456 |

C2 KV has 65536 payload bytes. Its index spec is padded from 8320 to 65536
bytes, placing 57216 unused bytes after the scales. SWA/state occupy offset
zero and are padded to their assigned slot capacity. Thirty SWA layers have
no padding; the ten layers in slot 3 have 16640 padding bytes per page. Each
state spec has 65536 payload bytes and is padded to 131072; G1 leaves slot 3
reserved but unused. Padding is applied to cloned specs and is idempotent.

With `N` global IDs (including the reserved null ID), the raw uint8 buffers
have sizes `N * [131072, 131072, 131072, 147712]`. Allocation, startup admission,
maximum-length sizing and concurrency use their sum: **540928 bytes =
528.25 KiB per global ID**. Merged C1/C2 block demand is counted once.

Typed zero-copy views retain the slot's physical page stride, not the
component's page size. C2 KV is `[N,64,1,512]` BF16; index K/scales are
`[N,64,1,128]` INT8 and `[N,64,1,1]` FP16. C1 uses the same widths with 128
rows. SWA is `[N,128,1,512]` BF16; state is `[N,16,1,1024]` FP32. Components
need not be contiguous; no whole-context gather is introduced by allocation.

### Earlier design comparisons

The original block-outermost implementation reserved 393216 bytes per ID
across 17 groups with separate C1/C2 groups, both using logical block 128.
The unimplemented comparison design kept C2 logical block 256 (128 stored
rows), C1 block 128 and 17 groups sharing three 147712-byte slots: 432.75 KiB
per ID. The new design reserves 528.25 KiB per ID but reduces the number of
full-context and SWA-group IDs. Compare memory for the same request workload,
including state retention and free/null IDs, rather than comparing the
per-ID divisor alone. Operator and serving performance remain unmeasured.

Index source layers 24, 28, 32 and 36 compute new selections in the reference
architecture but do not own another copy of the long KV or Index K. Candidate
blocks originate at layer 20. Consumers retain the source prefix and retrieve
the source cache from `static_forward_context`; shared modules are never
re-registered under consumer layers.

## Supported milestone and remaining accuracy work

The runtime contract is model runner V1, eager mode, BF16 cache, hybrid KV
management, PP/DCP/PCP equal to one, and tensor/data/expert parallel serving.
Prefix caching, speculative decoding, KV transfer and graph mode fail closed.

The fallback attends over local SWA plus the compressed rows selected by the
Indexer/Candidate path. Engram execution is intentionally disabled: its two
roughly 196 GB embedding tables require a distributed HBM layout, while the
temporary CPU/NFS mmap implementation was both prohibitively slow and
numerically unverified. Full accuracy still requires HBM-sharded Engram at
layers 1 and 14 and reference FP8/FP4 rounding. These omissions must not be
interpreted as full model accuracy.

## Validation

For the earlier block-outermost implementation, on the A3 remote container
the W8A8 checkpoint loads under TP4/DP4/EP, the
service becomes healthy, and greedy eager smoke requests return coherent
answers (`2+2 -> 4`, Chinese capital question -> Beijing). The focused cache and
mHC suite passes 30 tests. Formal task-level and long-context accuracy remain
follow-up gates.

The merged four-slot implementation has local static validation only.
Allocator/worker/metadata tests and slot-backed QLI/SparseFlashMla tests are
provided, but their torch/NPU execution is deferred. The earlier serving
results above do not validate this layout. Remote synchronization, builds,
correctness verification and performance measurements require a later run.
