# Autotune locally: cp submission-local.py submission.py && python ../eval.py benchmark .
# Paste best configs into submission.py before submitting to KernelBot.

from task import input_t, output_t

import torch
import helion
import helion.language as hl
from pathlib import Path

# ACF search: use /opt/booster_pack/
_BOOSTER_DIR = Path("/opt/booster_pack")
acf_files = sorted(str(p) for p in _BOOSTER_DIR.glob("fp8_group_quant_*.acf")) if _BOOSTER_DIR.exists() else []
_kernel_kwargs = {"static_shapes": True, "autotune_effort": "quick"}
if acf_files:
    _kernel_kwargs["autotune_search_acf"] = acf_files

# Placeholder configs for autotuner override. Paste best configs into submission.py before submitting.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    (1, 256, 64): helion.Config(block_sizes=[1], num_warps=1, num_stages=1),
    (4, 512, 128): helion.Config(block_sizes=[1], num_warps=1, num_stages=1),
    (16, 1024, 64): helion.Config(block_sizes=[1], num_warps=1, num_stages=1),
    (1, 4096, 128): helion.Config(block_sizes=[1], num_warps=1, num_stages=1),
    (8, 4096, 128): helion.Config(block_sizes=[1], num_warps=1, num_stages=1),
    (16, 4096, 128): helion.Config(block_sizes=[1], num_warps=1, num_stages=1),
    (256, 4096, 128): helion.Config(block_sizes=[1], num_warps=1, num_stages=1),
    (256, 8192, 128): helion.Config(block_sizes=[1], num_warps=1, num_stages=1),
    (4096, 7168, 128): helion.Config(block_sizes=[1], num_warps=1, num_stages=1),
}


def _make_kernel(config: helion.Config):
    @helion.kernel(**_kernel_kwargs)
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
