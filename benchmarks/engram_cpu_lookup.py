# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU lookup/dequant component benchmark; excludes routing and H2D."""

import argparse

import torch
from torch.utils.benchmark import Timer

from vllm_ascend import vllm_ascend_C  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    weight = torch.randint(-128, 128, (131072, 256), dtype=torch.int8)
    scale = torch.rand(131072, 8)
    for count in (1, 8, 32, 128, 512, 2048):
        ids = torch.randint(131072, (count,))
        output = torch.empty(count, 256, dtype=torch.bfloat16)
        reference = (weight[ids].float().reshape(-1, 8, 32) * scale[ids, :, None]).flatten(1).bfloat16()
        torch.ops._C_ascend.engram_int8_lookup_cpu(weight, scale, ids, output)
        assert torch.equal(output.view(torch.int16), reference.view(torch.int16))
        for name, statement in (
            ("fused", "torch.ops._C_ascend.engram_int8_lookup_cpu(weight, scale, ids, output)"),
            ("torch", "output.copy_((weight[ids].float().reshape(-1, 8, 32) * scale[ids, :, None]).flatten(1))"),
        ):
            result = Timer(statement, globals={**globals(), **locals()}, num_threads=args.threads).blocked_autorange()
            print(f"rows={count} {name}: median_us={result.median * 1e6:.2f}")


if __name__ == "__main__":
    main()
