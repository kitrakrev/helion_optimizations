#!POPCORN leaderboard gated_deltanet_recompute_w_u
#!POPCORN gpu B200_Nebius

# TileIR + TF32 + exp2 fast-math. ACFs not compatible with TileIR.
import os
os.environ["ENABLE_TILE"] = "1"
os.environ["HELION_BACKEND"] = "tileir"

from task import input_t, output_t

import torch
import helion
import helion.language as hl

# log2(e) for exp2 fast-math: e^x = 2^(x * log2(e))
LOG2_E = 1.4426950408889634

# TF32 + warp tuning: 8 warps for K/V=128 (larger blocks), 4 for K/V=64
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[], num_warps=8, num_stages=2),
    (1, 64, 1, 64, 64): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
    (2, 512, 3, 64, 64): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
    (2, 1024, 3, 64, 64): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
    (3, 1024, 4, 100, 100): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
    (4, 1024, 4, 128, 128): helion.Config(block_sizes=[], num_warps=8, num_stages=2),
    (2, 1536, 4, 128, 128): helion.Config(block_sizes=[], num_warps=8, num_stages=2),
    (4, 2048, 8, 64, 64): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="tf32", config=config)
    def kernel(k, v, beta, A, g):
        B, T, H, K = k.shape
        V = v.shape[-1]
        C = hl.specialize(A.shape[-1])
        K = hl.specialize(K)
        V = hl.specialize(V)
        w_out = torch.empty_like(k)
        u_out = torch.empty_like(v)
        BH = B * H
        for flat_bh, rt in hl.tile([BH, T], block_size=[1, C]):
            b_idx = flat_bh.begin // H
            h_idx = flat_bh.begin % H
            # Inputs are float32 per task.yml; avoid redundant casts
            A_chunk = A[b_idx, rt, h_idx, :]
            k_chunk = k[b_idx, rt, h_idx, :]
            v_chunk = v[b_idx, rt, h_idx, :]
            beta_chunk = beta[b_idx, rt, h_idx]
            g_chunk = g[b_idx, rt, h_idx]
            # exp2 fast-math: e^x = 2^(x * log2(e))
            decay = torch.exp2(g_chunk * LOG2_E)
            scaled_k = k_chunk * (beta_chunk * decay)[:, None]
            scaled_v = v_chunk * beta_chunk[:, None]
            w_chunk = hl.dot(A_chunk, scaled_k, out_dtype=torch.float32)
            u_chunk = hl.dot(A_chunk, scaled_v, out_dtype=torch.float32)
            w_out[b_idx, rt, h_idx, :] = w_chunk.to(k.dtype)
            u_out[b_idx, rt, h_idx, :] = u_chunk.to(v.dtype)
        return w_out, u_out
    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    return _KERNELS[(B, T, H, K, V)](k, v, beta, A, g)
