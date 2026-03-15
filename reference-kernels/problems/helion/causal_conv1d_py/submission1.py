#!POPCORN leaderboard causal_conv1d
#!POPCORN gpu B200_Nebius

# TileIR + Gemini fixes: config block_sizes in tile, implicit padding (no F.pad)
import os
os.environ["ENABLE_TILE"] = "1"
os.environ["HELION_BACKEND"] = "tileir"

from task import input_t, output_t

import torch
import helion
import helion.language as hl

# Optimized for B200: Larger S-tiles, 4-8 warps for better occupancy, and 3-4 stages to hide latency
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    (1, 64, 64, 4): helion.Config(block_sizes=[1, 64, 64], num_warps=4, num_stages=4),
    (2, 128, 128, 4): helion.Config(block_sizes=[1, 64, 128], num_warps=4, num_stages=4),
    (1, 256, 256, 3): helion.Config(block_sizes=[1, 64, 128], num_warps=8, num_stages=4),
    (1, 128, 64, 8): helion.Config(block_sizes=[1, 64, 64], num_warps=4, num_stages=4),
    (4, 64, 128, 4): helion.Config(block_sizes=[1, 64, 128], num_warps=4, num_stages=4),
    (1, 768, 512, 4): helion.Config(block_sizes=[1, 128, 128], num_warps=8, num_stages=4),
    (1, 768, 2048, 4): helion.Config(block_sizes=[1, 128, 256], num_warps=8, num_stages=3),
    (1, 1536, 2048, 4): helion.Config(block_sizes=[1, 128, 256], num_warps=8, num_stages=3),
    (1, 2560, 2048, 4): helion.Config(block_sizes=[1, 64, 256], num_warps=8, num_stages=3),
    (1, 2560, 4096, 4): helion.Config(block_sizes=[1, 64, 256], num_warps=8, num_stages=3),
}

def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def kernel(
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        B = x.size(0)
        D = x.size(1)
        S = x.size(2)
        W = hl.specialize(w.size(1))
        y = torch.empty(B, D, S, dtype=x.dtype, device=x.device)
        
        for rb, rd, rs in hl.tile([B, D, S]):
            bi = rb.begin
            
            # Hoist bias memory load out of inner ops
            bias_val = b[rd].to(torch.float32)
            acc = hl.zeros([rd, rs], dtype=torch.float32)
            
            for j in range(W):
                c = w[rd, j].to(torch.float32)
                seq_idx = rs.index + j - (W - 1)
                x_val = hl.load(x, [bi, rd, seq_idx], extra_mask=(seq_idx >= 0)).to(torch.float32)
                acc = acc + x_val * c[:, None]
                
            acc = acc + bias_val[:, None]
            y[rb, rd, rs] = acc[None, :, :].to(y.dtype)
            
        return y

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    x, weight, bias = data
    B, D, S = x.shape
    W = weight.shape[1]
    kernel = _KERNELS[(B, D, S, W)]
    return kernel(x, weight, bias)