"""Autotune gated_deltanet_chunk_fwd_h with ACFs on all shapes."""
import os, sys
os.environ["HELION_AUTOTUNE_PRECOMPILE"] = "spawn"

sys.path.insert(0, '.')
sys.path.insert(0, 'gated_deltanet_chunk_fwd_h_py')

import torch
import helion
import helion.language as hl
from pathlib import Path

ACF_FILES = sorted(str(p) for p in Path('/opt/booster_pack').glob('chunk_fwd_h_*.acf'))

# All test + benchmark shapes: (B, T, H, K, V)
ALL_SHAPES = [
    # Test
    (1, 64, 2, 64, 64),
    (2, 128, 4, 64, 64),
    (1, 256, 4, 64, 128),
    # Benchmark
    (1, 64, 1, 64, 64),
    (2, 512, 3, 64, 64),
    (2, 1024, 3, 64, 64),
    (3, 1024, 4, 100, 100),
    (4, 1024, 4, 128, 128),
    (2, 1536, 4, 128, 128),
    (4, 2048, 8, 64, 64),
]

# Skip already tuned shapes if resuming
SKIP = set()
if len(sys.argv) > 1:
    SKIP = set(int(x) for x in sys.argv[1].split(','))

CHUNK_SIZE = 64

def gen_inputs(B, T, H, K, V):
    torch.manual_seed(42)
    device = 'cuda'
    k = torch.randn(B, T, H, K, dtype=torch.float32, device=device) / K**0.5
    w = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    u = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    g_inc = -torch.abs(torch.randn(B, T, H, dtype=torch.float32, device=device))
    g = g_inc.cumsum(dim=1)
    g_cumsum = g.float().reshape(B, T // CHUNK_SIZE, CHUNK_SIZE, H).cumsum(dim=2).reshape(B, T, H)
    return k, w, u, g_cumsum


def make_kernel():
    @helion.kernel(static_shapes=True, dot_precision='ieee',
                   autotune_effort='quick', autotune_search_acf=ACF_FILES)
    def kernel(
        k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H, K = k.shape
        V = u.shape[-1]
        C = 64
        K = hl.specialize(K)
        V = hl.specialize(V)
        NT = (T + C - 1) // C
        h_out = torch.empty(B, NT, H, K, V, dtype=k.dtype, device=k.device)
        v_out = torch.empty_like(u)
        BH = B * H
        for flat, tv in hl.tile([BH, V], block_size=[1, None]):
            b_idx = flat.begin // H
            h_idx = flat.begin % H
            state = hl.zeros([K, tv], dtype=torch.float32)
            for tc in hl.tile(T, block_size=C):
                chunk_idx = tc.begin // C
                t_end = min(tc.begin + C, T) - 1
                h_out[b_idx, chunk_idx, h_idx, :, tv] = state.to(k.dtype)
                proj = hl.dot(w[b_idx, tc, h_idx, :], state, out_dtype=torch.float32)
                diff = u[b_idx, tc, h_idx, tv].to(torch.float32) - proj
                v_out[b_idx, tc, h_idx, tv] = diff.to(u.dtype)
                g_end = g[b_idx, t_end, h_idx].to(torch.float32)
                g_t = g[b_idx, tc, h_idx].to(torch.float32)
                valid = tc.index < T
                alpha = torch.where(valid, torch.exp(g_end - g_t), 0.0)
                k_adj = k[b_idx, tc, h_idx, :] * alpha[:, None]
                state = state * torch.exp(g_end)
                state = state + hl.dot(k_adj.T, diff, out_dtype=torch.float32)
        return h_out, v_out
    return kernel


if __name__ == "__main__":
    print("Autotuning gated_deltanet_chunk_fwd_h with ACFs...")
    for idx, shape in enumerate(ALL_SHAPES):
        if idx in SKIP:
            continue
        B, T, H, K, V = shape
        print(f"\n[{idx}] Autotuning shape {shape}...", flush=True)
        k, w, u, g = gen_inputs(B, T, H, K, V)
        kernel = make_kernel()
        try:
            result = kernel(k, w, u, g)
            print(f"  [DONE] Shape {shape} completed", flush=True)
        except Exception as e:
            print(f"  [ERROR] Shape {shape}: {e}", flush=True)
    print("\nAll done.")
