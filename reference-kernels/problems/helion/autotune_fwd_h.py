"""Autotune script for gated_deltanet_chunk_fwd_h per shape."""
import os
os.environ["ENABLE_TILE"] = "1"
os.environ["HELION_BACKEND"] = "tileir"
os.environ["HELION_AUTOTUNE_PRECOMPILE"] = "spawn"

import sys
os.chdir("gated_deltanet_chunk_fwd_h_py")
sys.path.insert(0, os.getcwd())

import torch
import helion
import helion.language as hl
from reference import generate_input

ACF_FILES = [
    "/opt/booster_pack/chunk_fwd_h_0.acf",
    "/opt/booster_pack/chunk_fwd_h_1.acf",
]

# Benchmark shapes to tune
SHAPES = [
    (1, 64, 1, 64, 64),
    (2, 512, 3, 64, 64),
    (2, 1024, 3, 64, 64),
    (3, 1024, 4, 100, 100),
    (4, 1024, 4, 128, 128),
    (2, 1536, 4, 128, 128),
    (4, 2048, 8, 64, 64),
]


def make_kernel():
    @helion.kernel(static_shapes=True, dot_precision="ieee",
                   autotune_search_acf=ACF_FILES)
    def kernel(
        k: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        g: torch.Tensor,
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

                proj = hl.dot(
                    w[b_idx, tc, h_idx, :], state, out_dtype=torch.float32
                )
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
    print("Starting autotune for gated_deltanet_chunk_fwd_h...")
    for shape in SHAPES:
        B, T, H, K, V = shape
        print(f"\nAutotuning shape {shape}...")
        data = generate_input(B, T, H, K, V, seed=42)
        k, w, u, g = data
        kernel = make_kernel()
        result = kernel(k, w, u, g)
        best_config = kernel.best_config
        print(f"  Best config for {shape}: {best_config}")
