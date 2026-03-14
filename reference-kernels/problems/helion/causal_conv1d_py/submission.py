#!POPCORN leaderboard causal_conv1d
#!POPCORN gpu B200_Nebius

from task import input_t, output_t

import base64
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
import helion
import helion.language as hl


# Embedded ACF (base64-encoded /opt/booster_pack/causal_conv_0.acf)
_ACF_B64 = "dxWiJeYWC+SCK/5QhmxuRcanuceCqhxDM2nM22IiPuEVMdEW4UsGPBXBUMu4/UPM2bcfwMCYOIfjwozWOpc2Zyd19oqHn41BHWmTagQFzmp34gPFFkoV0rlSm5OvbIku9Ipqu5kaiVuVNmvKjv6i6vBlDwFCE7qmj3czug6VvwfRZOgztYOXoFSKOZSxkcxOd6jJ9FL/Sqxc78ch+dSTrB9C2YdUEPW/cZv079l02J2EcZOAg1O6vFP4LQ=="


def _get_acf_path():
    if hasattr(_get_acf_path, "_path"):
        return _get_acf_path._path
    # Use local file if available, otherwise decode embedded
    local = Path("/opt/booster_pack/causal_conv_0.acf")
    if local.exists():
        _get_acf_path._path = str(local)
    else:
        d = tempfile.mkdtemp(prefix="causal_acf_")
        p = Path(d) / "causal_conv_0.acf"
        p.write_bytes(base64.b64decode(_ACF_B64))
        _get_acf_path._path = str(p)
    return _get_acf_path._path


_ACF = _get_acf_path()

# Per-shape ACF-optimized configs from autotuning on B200.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 64, 4): helion.Config(advanced_controls_file=_ACF, block_sizes=[32, 16], indexing=['pointer', 'pointer', 'pointer', 'pointer'], l2_groupings=[1], load_eviction_policies=['', 'last', ''], loop_orders=[[0, 1, 2]], num_sm_multiplier=1, num_stages=3, num_warps=16, pid_type='persistent_interleaved', range_flattens=[None, None], range_multi_buffers=[None, True], range_num_stages=[0, 1], range_unroll_factors=[0, 0], range_warp_specializes=[None, None], static_ranges=[False]),
    (2, 128, 128, 4): helion.Config(advanced_controls_file=_ACF, block_sizes=[16, 16], indexing=['pointer', 'pointer', 'pointer', 'pointer'], l2_groupings=[1], load_eviction_policies=['', '', ''], loop_orders=[[0, 1, 2]], num_stages=1, num_warps=32, pid_type='flat', range_flattens=[None, False], range_multi_buffers=[None, None], range_num_stages=[0, 0], range_unroll_factors=[0, 0], range_warp_specializes=[None, True], static_ranges=[False]),
    (1, 256, 256, 3): helion.Config(advanced_controls_file=_ACF, block_sizes=[16, 32], indexing=['pointer', 'pointer', 'pointer', 'pointer'], l2_groupings=[1], load_eviction_policies=['', '', ''], loop_orders=[[0, 2, 1]], num_stages=1, num_warps=16, pid_type='flat', range_flattens=[None, None], range_multi_buffers=[None, None], range_num_stages=[0, 0], range_unroll_factors=[0, 0], range_warp_specializes=[None, None], static_ranges=[True]),
    (1, 128, 64, 8): helion.Config(advanced_controls_file=_ACF, block_sizes=[32, 16], indexing=['pointer', 'tensor_descriptor', 'pointer', 'pointer'], l2_groupings=[1], load_eviction_policies=['', '', ''], loop_orders=[[0, 1, 2]], num_stages=1, num_warps=32, pid_type='flat', range_flattens=[None, None], range_multi_buffers=[None, None], range_num_stages=[0, 0], range_unroll_factors=[0, 0], range_warp_specializes=[None, None], static_ranges=[False]),
    (4, 64, 128, 4): helion.Config(advanced_controls_file=_ACF, block_sizes=[16, 64], indexing=['pointer', 'pointer', 'pointer', 'pointer'], l2_groupings=[1], load_eviction_policies=['last', 'last', ''], loop_orders=[[0, 1, 2]], num_stages=1, num_warps=16, pid_type='flat', range_flattens=[None, True], range_multi_buffers=[None, None], range_num_stages=[0, 0], range_unroll_factors=[0, 0], range_warp_specializes=[None, None], static_ranges=[False]),
    # Benchmark shapes
    (1, 768, 512, 4): helion.Config(advanced_controls_file=_ACF, block_sizes=[32, 32], indexing=['pointer', 'pointer', 'pointer', 'pointer'], l2_groupings=[1], load_eviction_policies=['', '', ''], loop_orders=[[0, 1, 2]], num_stages=1, num_warps=4, pid_type='flat', range_flattens=[None, None], range_multi_buffers=[None, None], range_num_stages=[0, 0], range_unroll_factors=[0, 0], range_warp_specializes=[None, None], static_ranges=[False]),
    (1, 768, 2048, 4): helion.Config(advanced_controls_file=_ACF, block_sizes=[32, 32], indexing=['pointer', 'pointer', 'pointer', 'pointer'], l2_groupings=[1], load_eviction_policies=['', '', ''], loop_orders=[[0, 1, 2]], num_stages=1, num_warps=4, pid_type='flat', range_flattens=[None, None], range_multi_buffers=[None, None], range_num_stages=[0, 0], range_unroll_factors=[0, 0], range_warp_specializes=[None, None], static_ranges=[False]),
    (1, 1536, 2048, 4): helion.Config(advanced_controls_file=_ACF, block_sizes=[32, 32], indexing=['pointer', 'pointer', 'pointer', 'pointer'], l2_groupings=[1], load_eviction_policies=['', 'last', ''], loop_orders=[[0, 1, 2]], num_stages=2, num_warps=4, pid_type='flat', range_flattens=[None, None], range_multi_buffers=[None, None], range_num_stages=[0, 0], range_unroll_factors=[0, 1], range_warp_specializes=[None, None], static_ranges=[False]),
    (1, 2560, 2048, 4): helion.Config(advanced_controls_file=_ACF, block_sizes=[8, 128], indexing=['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], l2_groupings=[1], load_eviction_policies=['', 'first', ''], loop_orders=[[0, 1, 2]], num_stages=1, num_warps=1, pid_type='flat', range_flattens=[None, None], range_multi_buffers=[None, None], range_num_stages=[0, 0], range_unroll_factors=[0, 0], range_warp_specializes=[None, None], static_ranges=[False]),
    (1, 2560, 4096, 4): helion.Config(advanced_controls_file=_ACF, block_sizes=[8, 128], indexing=['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], l2_groupings=[1], load_eviction_policies=['', 'first', ''], loop_orders=[[0, 1, 2]], num_stages=1, num_warps=1, pid_type='flat', range_flattens=[None, None], range_multi_buffers=[None, None], range_num_stages=[0, 0], range_unroll_factors=[0, 0], range_warp_specializes=[None, None], static_ranges=[False]),
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
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


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    x, weight, bias = data
    B, D, S = x.shape
    W = weight.shape[1]
    kernel = _KERNELS[(B, D, S, W)]
    padded = F.pad(x, (W - 1, 0))
    return kernel(padded, weight, bias)
