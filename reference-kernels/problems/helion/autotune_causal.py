"""Autotune causal_conv1d benchmark shapes, print best configs."""
import os, sys
os.environ["HELION_AUTOTUNE_PRECOMPILE"] = "spawn"
os.chdir("causal_conv1d_py")
sys.path.insert(0, os.getcwd())

import torch
import torch.nn.functional as F
import helion
import helion.language as hl
from reference import generate_input

BENCHMARK_SHAPES = [
    (1, 768, 512, 4),
    (1, 768, 2048, 4),
    (1, 1536, 2048, 4),
    (1, 2560, 2048, 4),
    (1, 2560, 4096, 4),
]


def make_kernel():
    @helion.kernel(static_shapes=True, autotune_effort="quick")
    def kernel(
        x_pad: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        B = x_pad.size(0)
        D = x_pad.size(1)
        L = x_pad.size(2)
        W = hl.specialize(w.size(1))
        N = L - W + 1
        y = torch.empty(B, D, N, dtype=x_pad.dtype, device=x_pad.device)
        for rb, rd, rs in hl.tile([B, D, N], block_size=[1, None, None]):
            bi = rb.begin
            acc = hl.zeros([rd, rs], dtype=torch.float32)
            for j in range(W):
                c = w[rd, j].to(torch.float32)
                x_val = hl.load(x_pad, [bi, rd, rs.index + j]).to(torch.float32)
                acc = acc + x_val * c[:, None]
            acc = acc + b[rd].to(torch.float32)[:, None]
            y[rb, rd, rs] = acc[None, :, :].to(y.dtype)
        return y
    return kernel


if __name__ == "__main__":
    print("Autotuning causal_conv1d benchmark shapes...")
    for shape in BENCHMARK_SHAPES:
        B, D, S, W = shape
        print(f"\nShape {shape}...")
        x, weight, bias = generate_input(B, D, S, W, seed=42)
        padded = F.pad(x, (W - 1, 0))
        kernel = make_kernel()
        kernel(padded, weight, bias)
        print(f"  Best config: {kernel.best_config}")
    print("\nDone.")
