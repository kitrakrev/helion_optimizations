#!POPCORN leaderboard gated_deltanet_chunk_fwd_o
#!POPCORN gpu B200_Nebius

# Autotune locally: cp submission-local.py submission.py && python ../eval.py benchmark .
# Paste best configs into submission.py before submitting to KernelBot.

from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Placeholder configs for autotuner override. Paste best configs into submission.py before submitting.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[], num_warps=1, num_stages=1),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[], num_warps=1, num_stages=1),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[], num_warps=1, num_stages=1),
    # Benchmark shapes
    (1, 64, 1, 64, 64): helion.Config(block_sizes=[], num_warps=1, num_stages=1),
    (2, 512, 3, 64, 64): helion.Config(block_sizes=[], num_warps=1, num_stages=1),
    (2, 1024, 3, 64, 64): helion.Config(block_sizes=[], num_warps=1, num_stages=1),
    (3, 1024, 4, 100, 100): helion.Config(block_sizes=[], num_warps=1, num_stages=1),
    (4, 1024, 4, 128, 128): helion.Config(block_sizes=[], num_warps=1, num_stages=1),
    (2, 1536, 4, 128, 128): helion.Config(block_sizes=[], num_warps=1, num_stages=1),
    (4, 2048, 8, 64, 64): helion.Config(block_sizes=[], num_warps=1, num_stages=1),
}

# ACF search: use /opt/booster_pack/
from pathlib import Path

_BOOSTER_DIR = Path("/opt/booster_pack")
acf_files = sorted(str(p) for p in _BOOSTER_DIR.glob("chunk_fwd_o_*.acf")) if _BOOSTER_DIR.exists() else []
_kernel_kwargs = {"static_shapes": True, "dot_precision": "ieee", "autotune_effort": "quick"}
if acf_files:
    _kernel_kwargs["autotune_search_acf"] = acf_files


def _make_kernel(config: helion.Config):
    @helion.kernel(**_kernel_kwargs)
    def kernel(
        q: torch.Tensor,     # [B, T, H, K]
        k: torch.Tensor,     # [B, T, H, K]
        v: torch.Tensor,     # [B, T, H, V]
        h: torch.Tensor,     # [B, NT, H, K, V]
        g: torch.Tensor,     # [B, T, H]
        scale: float,
    ) -> torch.Tensor:
        B, T, H, K = q.shape
        V = v.shape[-1]
        C = 64
        K = hl.specialize(K)
        V = hl.specialize(V)

        out = torch.empty_like(v)

        BH = B * H
        for flat_bh, tile_t in hl.tile([BH, T], block_size=[None, C]):
            b_idx = flat_bh.begin // H
            h_idx = flat_bh.begin % H
            c_idx = tile_t.begin // C

            g_vals = g[b_idx, tile_t, h_idx]
            q_s = q[b_idx, tile_t, h_idx, :] * torch.exp(g_vals)[:, None]
            k_s = k[b_idx, tile_t, h_idx, :] * torch.exp(-g_vals)[:, None]

            sim1 = hl.dot(q_s, k_s.T)
            sim2 = hl.dot(q_s, k_s.T)
            sim = (sim1 + sim2) * 0.5
            idx = hl.arange(tile_t.block_size)
            mask = idx[:, None] >= idx[None, :]
            sim = torch.where(mask, sim, 0.0)
            local1 = hl.dot(sim.to(v.dtype), v[b_idx, tile_t, h_idx, :])
            local2 = hl.dot(sim.to(v.dtype), v[b_idx, tile_t, h_idx, :])
            local_out = (local1 + local2) * 0.5

            glob1 = hl.dot(q_s, h[b_idx, c_idx, h_idx, :, :])
            glob2 = hl.dot(q_s, h[b_idx, c_idx, h_idx, :, :])
            global_out = (glob1 + glob2) * 0.5

            out[b_idx, tile_t, h_idx, :] = ((global_out + local_out) * scale).to(out.dtype)

        return out

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    q, k, v_new, h, g = data
    B, T, H, K = q.shape
    V = v_new.shape[-1]
    scale = K ** -0.5
    kernel = _KERNELS[(B, T, H, K, V)]
    return kernel(q, k, v_new, h, g, scale)
