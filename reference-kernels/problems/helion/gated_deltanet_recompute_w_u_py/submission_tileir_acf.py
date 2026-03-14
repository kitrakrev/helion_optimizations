#!POPCORN leaderboard gated_deltanet_recompute_w_u
#!POPCORN gpu B200_Nebius

# EXPERIMENT: TileIR + ACF together. Test if compatible and faster.
import os
os.environ["ENABLE_TILE"] = "1"
os.environ["HELION_BACKEND"] = "tileir"

import base64
import tempfile
from pathlib import Path

from task import input_t, output_t

import torch
import helion
import helion.language as hl

_ACF_B64 = [
    "dxWiJeTK18brsHzCEqyUfsG2eoBthCEcv4NjjksNymWQxD8cU0RHVJJzMLQZiOz63XaZNYSLXok3djjkTO2n6Nn67r2HJaVwrjKLcuCbrgpqZqMuwKiGX7RRmC14HluHL1c4xLb3bK2LKIhXfpeVII+P1qia2+ItMFi4gE7VSFen9XmiJBIGMcUbNH5beyaknUIjHrgVoEa2BS3LEz55RvWoM22++qrKBO6BmqwBrejxBOb19ibPySaNYH339k8529dbfanIqYyHz3Kgc7KlAKfwXeM=",
]


def _ensure_acf():
    if hasattr(_ensure_acf, "_path"):
        return _ensure_acf._path
    d = tempfile.mkdtemp(prefix="recompute_w_u_acf_")
    p = Path(d) / "recompute_w_u_fwd_0.acf"
    p.write_bytes(base64.b64decode(_ACF_B64[0]))
    _ensure_acf._path = str(p)
    return _ensure_acf._path


_ACF_PATH = _ensure_acf()

SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    (1, 64, 2, 64, 64): helion.Config(advanced_controls_file=_ACF_PATH, block_sizes=[], num_warps=4, num_stages=2),
    (2, 128, 4, 64, 64): helion.Config(advanced_controls_file=_ACF_PATH, block_sizes=[], num_warps=4, num_stages=2),
    (1, 256, 4, 64, 128): helion.Config(advanced_controls_file=_ACF_PATH, block_sizes=[], num_warps=4, num_stages=2),
    (1, 64, 1, 64, 64): helion.Config(advanced_controls_file=_ACF_PATH, block_sizes=[], num_warps=4, num_stages=2),
    (2, 512, 3, 64, 64): helion.Config(advanced_controls_file=_ACF_PATH, block_sizes=[], num_warps=4, num_stages=2),
    (2, 1024, 3, 64, 64): helion.Config(advanced_controls_file=_ACF_PATH, block_sizes=[], num_warps=4, num_stages=2),
    (3, 1024, 4, 100, 100): helion.Config(advanced_controls_file=_ACF_PATH, block_sizes=[], num_warps=4, num_stages=2),
    (4, 1024, 4, 128, 128): helion.Config(advanced_controls_file=_ACF_PATH, block_sizes=[], num_warps=4, num_stages=2),
    (2, 1536, 4, 128, 128): helion.Config(advanced_controls_file=_ACF_PATH, block_sizes=[], num_warps=4, num_stages=2),
    (4, 2048, 8, 64, 64): helion.Config(advanced_controls_file=_ACF_PATH, block_sizes=[], num_warps=4, num_stages=2),
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
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
            A_chunk = A[b_idx, rt, h_idx, :].to(torch.float32)
            k_chunk = k[b_idx, rt, h_idx, :].to(torch.float32)
            v_chunk = v[b_idx, rt, h_idx, :].to(torch.float32)
            beta_chunk = beta[b_idx, rt, h_idx].to(torch.float32)
            g_chunk = g[b_idx, rt, h_idx].to(torch.float32)
            scaled_k = k_chunk * (beta_chunk * torch.exp(g_chunk))[:, None]
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
    return _KERNELS[(k.shape[0], k.shape[1], k.shape[2], k.shape[3], v.shape[-1])](k, v, beta, A, g)
