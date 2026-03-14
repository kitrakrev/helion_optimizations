"""Autotune gated_deltanet_chunk_fwd_o with ACFs on all shapes."""
import os, sys
os.environ["HELION_AUTOTUNE_PRECOMPILE"] = "spawn"

sys.path.insert(0, '.')
sys.path.insert(0, 'gated_deltanet_chunk_fwd_o_py')

import torch
import helion
import helion.language as hl
from pathlib import Path

ACF_FILES = sorted(str(p) for p in Path('/opt/booster_pack').glob('chunk_fwd_o_*.acf'))
print(f'ACF files: {ACF_FILES}')

SHAPES = [
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

def gen_inputs(B, T, H, K, V):
    torch.manual_seed(42)
    device = 'cuda'
    q = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    k = torch.randn(B, T, H, K, dtype=torch.float32, device=device) / K**0.5
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    h = torch.randn(B, T//64, H, K, V, dtype=torch.float32, device=device)
    g = -torch.abs(torch.randn(B, T, H, dtype=torch.float32, device=device)).cumsum(dim=1)
    g_cumsum = g.float().reshape(B, T//64, 64, H).cumsum(dim=2).reshape(B, T, H)
    return q, k, v, h, g_cumsum


def make_kernel():
    @helion.kernel(static_shapes=True, dot_precision='ieee',
                   autotune_effort='quick', autotune_search_acf=ACF_FILES)
    def kernel(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        h: torch.Tensor, g: torch.Tensor, scale: float,
    ) -> torch.Tensor:
        B, T, H, K = q.shape
        V = v.shape[-1]
        C = 64
        K = hl.specialize(K)
        V = hl.specialize(V)
        out = torch.empty_like(v)
        BH = B * H
        for flat_bh, tile_t in hl.tile([BH, T], block_size=[1, C]):
            b_idx = flat_bh.begin // H
            h_idx = flat_bh.begin % H
            c_idx = tile_t.begin // C
            g_c = g[b_idx, tile_t, h_idx].to(torch.float32)
            q_c = q[b_idx, tile_t, h_idx, :].to(torch.float32)
            k_c = k[b_idx, tile_t, h_idx, :].to(torch.float32)
            v_c = v[b_idx, tile_t, h_idx, :].to(torch.float32)
            h_c = h[b_idx, c_idx, h_idx, :, :].to(torch.float32)
            o_inter = hl.dot(q_c, h_c, out_dtype=torch.float32) * torch.exp(g_c)[:, None]
            qk = hl.dot(q_c, k_c.T, out_dtype=torch.float32)
            g_diff = g_c[:, None] - g_c[None, :]
            qk = qk * torch.exp(g_diff)
            causal = torch.tril(torch.ones(C, C, dtype=torch.float32, device=q.device))
            qk = qk * causal
            o_intra = hl.dot(qk, v_c, out_dtype=torch.float32)
            out[b_idx, tile_t, h_idx, :] = ((o_inter + o_intra) * scale).to(out.dtype)
        return out
    return kernel


if __name__ == "__main__":
    print("Autotuning gated_deltanet_chunk_fwd_o with ACFs...", flush=True)
    for idx, shape in enumerate(SHAPES):
        B, T, H, K, V = shape
        print(f"\n[{idx}] Shape {shape}...", flush=True)
        q, k, v, h, g = gen_inputs(B, T, H, K, V)
        scale = K ** -0.5
        kernel = make_kernel()
        try:
            result = kernel(q, k, v, h, g, scale)
            print(f"  [DONE] Shape {shape}", flush=True)
        except Exception as e:
            print(f"  [ERROR] Shape {shape}: {e}", flush=True)
    print("\nAll done.", flush=True)
