"""Parallel autotune causal_conv1d with wide search range across all shapes."""
import os
os.environ["ENABLE_TILE"] = "1"
os.environ["HELION_BACKEND"] = "tileir"
os.environ["HELION_AUTOTUNE_PRECOMPILE"] = "spawn"
os.environ["HELION_AUTOTUNE_PRECOMPILE_JOBS"] = "32"

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'causal_conv1d_py')

import torch
import helion
import helion.language as hl

# All shapes: (B, D, S, W)
SHAPES = [
    # Test
    (1, 64, 64, 4),
    (2, 128, 128, 4),
    (1, 256, 256, 3),
    (1, 128, 64, 8),
    (4, 64, 128, 4),
    # Benchmark
    (1, 768, 512, 4),
    (1, 768, 2048, 4),
    (1, 1536, 2048, 4),
    (1, 2560, 2048, 4),
    (1, 2560, 4096, 4),
]


def gen_inputs(B, D, S, W):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(42)
    x = torch.randn(B, D, S, dtype=torch.float32, device="cuda", generator=gen)
    weight = torch.randn(D, W, dtype=torch.float32, device="cuda", generator=gen)
    bias = torch.randn(D, dtype=torch.float32, device="cuda", generator=gen)
    return x, weight, bias


def make_kernel():
    @helion.kernel(static_shapes=True, autotune_effort="max")
    def kernel(
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        B = x.size(0)
        D = x.size(1)
        S = x.size(2)
        W = hl.specialize(w.size(1))
        y = torch.empty(B, D, S, dtype=x.dtype, device=x.device)
        for rb, rd, rs in hl.tile([B, D, S]):
            bi = rb.begin
            acc = hl.zeros([rd, rs], dtype=torch.float32)
            for j in range(W):
                c = w[rd, j].to(torch.float32)
                seq_idx = rs.index + j - (W - 1)
                x_val = hl.load(x, [bi, rd, seq_idx], extra_mask=(seq_idx >= 0)).to(torch.float32)
                acc = acc + x_val * c[:, None]
            acc = acc + b[rd].to(torch.float32)[:, None]
            y[rb, rd, rs] = acc[None, :, :].to(y.dtype)
        return y
    return kernel


if __name__ == "__main__":
    print(f"Autotuning causal_conv1d with 32 parallel jobs, effort=max...", flush=True)
    for idx, shape in enumerate(SHAPES):
        B, D, S, W = shape
        print(f"\n[{idx}] Shape {shape}...", flush=True)
        x, weight, bias = gen_inputs(B, D, S, W)
        kernel = make_kernel()
        try:
            result = kernel(x, weight, bias)
            print(f"  [DONE] Shape {shape}", flush=True)
        except Exception as e:
            print(f"  [ERROR] Shape {shape}: {e}", flush=True)
    print("\nAll done.", flush=True)
