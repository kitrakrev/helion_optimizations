"""Autotune causal_conv1d with ACFs on all shapes."""
import os, sys
os.environ["HELION_AUTOTUNE_PRECOMPILE"] = "spawn"

sys.path.insert(0, '.')
sys.path.insert(0, 'causal_conv1d_py')

import torch
import torch.nn.functional as F
import helion
import helion.language as hl
from pathlib import Path

ACF_FILES = sorted(str(p) for p in Path('/opt/booster_pack').glob('causal_conv_*.acf'))

SHAPES = [
    (1, 64, 64, 4),
    (2, 128, 128, 4),
    (1, 256, 256, 3),
    (1, 128, 64, 8),
    (4, 64, 128, 4),
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
    @helion.kernel(static_shapes=True, autotune_effort='quick',
                   autotune_search_acf=ACF_FILES)
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
    print("Autotuning causal_conv1d with ACFs...", flush=True)
    for idx, shape in enumerate(SHAPES):
        B, D, S, W = shape
        print(f"\n[{idx}] Shape {shape}...", flush=True)
        x, weight, bias = gen_inputs(B, D, S, W)
        padded = F.pad(x, (W - 1, 0))
        kernel = make_kernel()
        try:
            result = kernel(padded, weight, bias)
            print(f"  [DONE] Shape {shape}", flush=True)
        except Exception as e:
            print(f"  [ERROR] Shape {shape}: {e}", flush=True)
    print("\nAll done.", flush=True)
