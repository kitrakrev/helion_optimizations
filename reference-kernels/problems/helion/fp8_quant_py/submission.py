#!POPCORN leaderboard fp8_quant
#!POPCORN gpu B200_Nebius

# ACF files embedded as base64 (leaderboard env has no /opt/booster_pack). Decode to temp at load.
from task import input_t, output_t

import base64
import tempfile
from pathlib import Path

import torch
import helion
import helion.language as hl

# Embedded ACFs (base64) - from /opt/booster_pack/fp8_group_quant_*.acf
_ACF_B64 = {
    0: "dxWiJZMOCeaAKfxShG5sR8Slu8WAqB4ton4NP0t+6TsvOFjLmqhQ+XTzmP8lzv/wGB5xCIMgJgf9mXMpxWjJmNiKCXV4YHK+4pZslfs/",
    6: "dxWiJftmu1Qym07gNtze9XYXCXcyGhZJOWPG0WgoNEKojGyrXPa7gah87XYFQP5x8Z836OiwEK/LF+K4VPlYCUkbmOTp8eMvcwfQKUdGjSk0oSLALnIt6oFqo6uXVLEWzLJSg0fEV4VL6LUUUCAQWELXvbPwoQgUPcWBCLwn1buUa50fO6OWzQs6ZcRqRIitj2E+HhLI9WPWQaRLcwJFc37dWfC03MTeIsix5tTWuRkuxLIh/jrezCOp/bk/Qod+b4+H1gFjNGcyzyHDEM/9lVyYQhOZksLeIccXixC+a5X/596NI9T+OqHfn2qm3fXsELUpv19w05A3a1ACYFVX3gvos7h4bg==",
}


def _ensure_acf_paths():
    """Decode embedded ACFs to temp dir; return dict {0: path0, 6: path6}."""
    if hasattr(_ensure_acf_paths, "_paths"):
        return _ensure_acf_paths._paths
    d = tempfile.mkdtemp(prefix="fp8_quant_acf_")
    paths = {}
    for i, b64 in _ACF_B64.items():
        p = Path(d) / f"fp8_group_quant_{i}.acf"
        p.write_bytes(base64.b64decode(b64))
        paths[i] = str(p)
    _ensure_acf_paths._paths = paths
    return paths


_ACF = _ensure_acf_paths()
_ACF_0 = _ACF[0]
_ACF_6 = _ACF[6]

# Per-shape configs from autotuning. Use embedded ACFs for leaderboard (no /opt/booster_pack).
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 256, 64): helion.Config(advanced_controls_file=_ACF_0, block_sizes=[16], num_warps=8, num_stages=1),
    (4, 512, 128): helion.Config(advanced_controls_file=_ACF_0, block_sizes=[32], num_warps=8, num_stages=2),
    (16, 1024, 64): helion.Config(advanced_controls_file=_ACF_0, block_sizes=[16], num_warps=8, num_stages=1),
    (1, 4096, 128): helion.Config(advanced_controls_file=_ACF_0, block_sizes=[8], num_warps=2, num_stages=1),
    (8, 4096, 128): helion.Config(advanced_controls_file=_ACF_0, block_sizes=[32], num_warps=4, num_stages=1),
    # Benchmark shapes
    (16, 4096, 128): helion.Config(advanced_controls_file=_ACF_0, block_sizes=[1], num_warps=4, num_stages=1),
    (256, 4096, 128): helion.Config(advanced_controls_file=_ACF_6, block_sizes=[8], num_warps=4, num_stages=3),
    (256, 8192, 128): helion.Config(advanced_controls_file=_ACF_0, block_sizes=[8], num_warps=2, num_stages=1),
    (4096, 7168, 128): helion.Config(advanced_controls_file=_ACF_0, block_sizes=[32], num_warps=8, num_stages=1),
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def kernel(
        data: torch.Tensor,       # [N, G] input rows
        qout: torch.Tensor,       # [N, G] output buffer (writes in-place)
        scales_out: torch.Tensor,  # [N] output normalization factors
    ) -> None:
        nrows = data.size(0)
        ncols = hl.specialize(data.size(1))
        MAX_VAL = 448.0

        for rr in hl.tile(nrows):
            row = data[rr, :].to(torch.float32)
            # Triton escape hatch (~15%): fast warp-level amax reduction
            amax = hl.inline_triton(
                "tl.max(tl.abs({row}), axis=1)",
                args={"row": row},
                output_like=hl.zeros([rr.block_size], dtype=torch.float32),
            )
            amax = torch.clamp(amax, min=1e-10)
            scale = amax / MAX_VAL

            qout[rr, :] = torch.clamp(row / scale[:, None], min=-448.0, max=448.0)
            scales_out[rr] = scale

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    x, x_q, x_s = data
    T, H = x.shape
    G = x_s.shape[1]
    gsz = H // G
    N = T * G

    kernel = _KERNELS[(T, H, gsz)]

    flat_in = x.reshape(N, gsz)
    flat_q = x_q.reshape(N, gsz)
    flat_s = x_s.reshape(N)

    kernel(flat_in, flat_q, flat_s)
    return x_q, x_s
