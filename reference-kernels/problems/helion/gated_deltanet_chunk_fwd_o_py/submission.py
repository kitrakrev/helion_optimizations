#!POPCORN leaderboard gated_deltanet_chunk_fwd_o
#!POPCORN gpu B200_Nebius

from task import input_t, output_t

import torch
import helion
import helion.language as hl


SHAPE_CONFIGS: dict[tuple, helion.Config] = {}

_KERNELS: dict[tuple, object] = {}


def _make_kernel_default():
    @helion.kernel(static_shapes=True, dot_precision="ieee", autotune_effort="none")
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
        for flat_bh, tile_t in hl.tile([BH, T], block_size=[1, C]):
            b_idx = flat_bh.begin // H
            h_idx = flat_bh.begin % H
            c_idx = tile_t.begin // C

            g_c = g[b_idx, tile_t, h_idx].to(torch.float32)  # [C]
            q_c = q[b_idx, tile_t, h_idx, :].to(torch.float32)  # [C, K]
            k_c = k[b_idx, tile_t, h_idx, :].to(torch.float32)  # [C, K]
            v_c = v[b_idx, tile_t, h_idx, :].to(torch.float32)  # [C, V]
            h_c = h[b_idx, c_idx, h_idx, :, :].to(torch.float32)  # [K, V]

            # Inter-chunk: q @ h * exp(g)
            o_inter = hl.dot(q_c, h_c, out_dtype=torch.float32)  # [C, V]
            o_inter = o_inter * torch.exp(g_c)[:, None]

            # Intra-chunk: causal(q @ k^T * exp(g_i - g_j)) @ v
            qk = hl.dot(q_c, k_c.T, out_dtype=torch.float32)  # [C, C]
            g_diff = g_c[:, None] - g_c[None, :]  # [C, C]

            # Causal mask: avoid NaN from exp(large positive) * 0
            idx = hl.arange(tile_t.block_size)
            causal = (idx[:, None] >= idx[None, :]).to(torch.float32)
            # Clamp g_diff to 0 for upper triangle to prevent exp overflow
            g_diff_safe = torch.where(idx[:, None] >= idx[None, :], g_diff, 0.0)
            qk = qk * torch.exp(g_diff_safe) * causal

            o_intra = hl.dot(qk, v_c, out_dtype=torch.float32)  # [C, V]

            out[b_idx, tile_t, h_idx, :] = ((o_inter + o_intra) * scale).to(out.dtype)

        return out

    return kernel


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
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
        for flat_bh, tile_t in hl.tile([BH, T], block_size=[1, C]):
            b_idx = flat_bh.begin // H
            h_idx = flat_bh.begin % H
            c_idx = tile_t.begin // C

            g_c = g[b_idx, tile_t, h_idx].to(torch.float32)
            q_c = q[b_idx, tile_t, h_idx, :].to(torch.float32)
            k_c = k[b_idx, tile_t, h_idx, :].to(torch.float32)
            v_c = v[b_idx, tile_t, h_idx, :].to(torch.float32)
            h_c = h[b_idx, c_idx, h_idx, :, :].to(torch.float32)

            o_inter = hl.dot(q_c, h_c, out_dtype=torch.float32)
            o_inter = o_inter * torch.exp(g_c)[:, None]

            qk = hl.dot(q_c, k_c.T, out_dtype=torch.float32)
            g_diff = g_c[:, None] - g_c[None, :]

            idx = hl.arange(tile_t.block_size)
            causal = (idx[:, None] >= idx[None, :]).to(torch.float32)
            g_diff_safe = torch.where(idx[:, None] >= idx[None, :], g_diff, 0.0)
            qk = qk * torch.exp(g_diff_safe) * causal

            o_intra = hl.dot(qk, v_c, out_dtype=torch.float32)

            out[b_idx, tile_t, h_idx, :] = ((o_inter + o_intra) * scale).to(out.dtype)

        return out

    return kernel


def _get_kernel(shape_key):
    if shape_key not in _KERNELS:
        if shape_key in SHAPE_CONFIGS:
            _KERNELS[shape_key] = _make_kernel(SHAPE_CONFIGS[shape_key])
        else:
            _KERNELS[shape_key] = _make_kernel_default()
    return _KERNELS[shape_key]


def custom_kernel(data: input_t) -> output_t:
    q, k, v_new, h, g = data
    B, T, H, K = q.shape
    V = v_new.shape[-1]
    scale = K ** -0.5
    kernel = _get_kernel((B, T, H, K, V))
    return kernel(q, k, v_new, h, g, scale)
