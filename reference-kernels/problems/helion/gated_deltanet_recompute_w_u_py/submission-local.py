#!POPCORN leaderboard gated_deltanet_recompute_w_u
#!POPCORN gpu B200_Nebius

# Autotune: cp submission-local.py submission.py && python ../eval.py benchmark .
# TileIR backend (faster than Triton+ACF). ACFs not compatible.
import os
os.environ["ENABLE_TILE"] = "1"
os.environ["HELION_BACKEND"] = "tileir"

from task import input_t, output_t

import torch
import helion
import helion.language as hl

SHAPE_CONFIGS = {
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
    (1, 64, 1, 64, 64): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
    (2, 512, 3, 64, 64): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
    (2, 1024, 3, 64, 64): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
    (3, 1024, 4, 100, 100): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
    (4, 1024, 4, 128, 128): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
    (2, 1536, 4, 128, 128): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
    (4, 2048, 8, 64, 64): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
}


def _make_kernel(config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config, autotune_effort="quick")
    def kernel(k, v, beta, A, g):
        B, T, H, K = k.shape
        V = v.shape[-1]
        C = hl.specialize(A.shape[-1])
        K, V = hl.specialize(K), hl.specialize(V)
        w_out, u_out = torch.empty_like(k), torch.empty_like(v)
        BH = B * H
        for flat_bh, rt in hl.tile([BH, T], block_size=[1, C]):
            b_idx, h_idx = flat_bh.begin // H, flat_bh.begin % H
            A_chunk = A[b_idx, rt, h_idx, :].to(torch.float32)
            k_chunk = k[b_idx, rt, h_idx, :].to(torch.float32)
            v_chunk = v[b_idx, rt, h_idx, :].to(torch.float32)
            beta_chunk = beta[b_idx, rt, h_idx].to(torch.float32)
            g_chunk = g[b_idx, rt, h_idx].to(torch.float32)
            scaled_k = k_chunk * (beta_chunk * torch.exp(g_chunk))[:, None]
            scaled_v = v_chunk * beta_chunk[:, None]
            w_out[b_idx, rt, h_idx, :] = hl.dot(A_chunk, scaled_k, out_dtype=torch.float32).to(k.dtype)
            u_out[b_idx, rt, h_idx, :] = hl.dot(A_chunk, scaled_v, out_dtype=torch.float32).to(v.dtype)
        return w_out, u_out
    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    return _KERNELS[(k.shape[0], k.shape[1], k.shape[2], k.shape[3], v.shape[-1])](k, v, beta, A, g)
