#!/usr/bin/env python3
"""Autotune causal_conv1d (TileIR) and print best configs for submission.py.

Usage:
  cd reference-kernels/problems/helion
  ENABLE_TILE=1 HELION_BACKEND=tileir python causal_conv1d_py/autotune.py [--effort quick|full]

Output: Prints helion.Config(...) lines ready to paste into SHAPE_CONFIGS.
"""
import argparse
import os
import sys
from pathlib import Path

# Must set BEFORE importing torch (which loads triton)
os.environ["ENABLE_TILE"] = "1"
os.environ["HELION_BACKEND"] = "tileir"
os.environ.setdefault("HELION_AUTOTUNE_PRECOMPILE", "spawn")

_script_dir = Path(__file__).resolve().parent
_helion_dir = _script_dir.parent
os.chdir(_script_dir)
sys.path.insert(0, str(_helion_dir))
sys.path.insert(0, str(_script_dir))

import torch  # noqa: E402
import helion
import helion.language as hl
from reference import generate_input

ALL_SHAPES = [
    (1, 64, 64, 4),
    (2, 128, 128, 4),
    (1, 256, 256, 3),
    (1, 128, 64, 8),
    (4, 64, 128, 4),
    (1, 768, 512, 4),
    (1, 768, 2048, 4),
    (1, 1536, 2048, 4),
    (1, 2560, 2048, 4),
    (1, 2560, 4096, 4),
]


def make_autotune_kernel(effort: str):
    """Kernel with autotune - matches submission (implicit padding, no F.pad)."""

    @helion.kernel(static_shapes=True, autotune_effort=effort)
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
            acc = hl.zeros([rd, rs], dtype=torch.float32)
            for j in range(W):
                c = w[rd, j].to(torch.float32)
                seq_idx = rs.index + j - (W - 1)
                x_val = hl.load(x, [bi, rd, seq_idx], extra_mask=(seq_idx >= 0)).to(torch.float32)
                acc = acc + x_val * c[:, None]
            acc = acc + b[rd].to(torch.float32)[:, None]
            y[rb, rd, rs] = acc[None, :, :].to(y.dtype)
        return y

    return kernel


def config_to_submission_str(config: helion.Config) -> str:
    """Format config for submission.py - TileIR uses block_sizes, num_warps, num_stages."""
    cfg = config.config
    parts = []
    for k in ["block_sizes", "num_warps", "num_stages"]:
        if k in cfg:
            parts.append(f"{k}={repr(cfg[k])}")
    return "helion.Config(" + ", ".join(parts) + ")"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--effort", default="full", choices=["quick", "full"])
    ap.add_argument("--shape", default=None, help="Comma-separated shape indices (default: all)")
    args = ap.parse_args()

    shape_indices = range(len(ALL_SHAPES))
    if args.shape is not None:
        shape_indices = [int(x.strip()) for x in args.shape.split(",")]

    print(f"TileIR causal_conv1d autotune, effort={args.effort}", flush=True)
    print(f"Shapes: {[ALL_SHAPES[i] for i in shape_indices]}", flush=True)

    results = {}
    for idx in shape_indices:
        shape = ALL_SHAPES[idx]
        B, D, S, W = shape
        print(f"\n[{idx}] Autotuning shape {shape}...", flush=True)
        x, weight, bias = generate_input(B, D, S, W, seed=42)
        kernel_fn = make_autotune_kernel(args.effort)
        bound = kernel_fn.bind((x, weight, bias))
        try:
            best_config = bound.autotune((x, weight, bias), force=True)
            results[shape] = best_config
            cfg_str = config_to_submission_str(best_config)
            print(f"  [DONE] {cfg_str}", flush=True)
        except Exception as e:
            print(f"  [ERROR] {e}", flush=True)
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("SHAPE_CONFIGS (paste into submission.py):")
    print("=" * 60)
    for shape, cfg in results.items():
        cfg_str = config_to_submission_str(cfg)
        print(f"    {shape}: {cfg_str},")


if __name__ == "__main__":
    main()
