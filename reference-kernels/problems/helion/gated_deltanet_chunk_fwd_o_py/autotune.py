#!/usr/bin/env python3
"""Autotune gated_deltanet_chunk_fwd_o and print best configs for submission.py.

Usage:
  cd reference-kernels/problems/helion
  python gated_deltanet_chunk_fwd_o_py/autotune.py [--effort quick|full] [--shape 0,1,2,...]

Output: Prints helion.Config(...) lines ready to paste into SHAPE_CONFIGS.
"""
import argparse
import base64
import os
import sys
import tempfile
from pathlib import Path

# Must set before importing helion
os.environ.setdefault("HELION_AUTOTUNE_PRECOMPILE", "spawn")

# Add paths and chdir like eval.py (utils.py is in helion dir)
_script_dir = Path(__file__).resolve().parent
_helion_dir = _script_dir.parent
os.chdir(_script_dir)
sys.path.insert(0, str(_helion_dir))  # for utils
sys.path.insert(0, str(_script_dir))  # for reference, task

import torch
import helion
import helion.language as hl
from reference import generate_input

LOG2_E = 1.4426950408889634

# Embedded ACF (chunk_fwd_o_2.acf) - fallback when /opt/booster_pack not available
_ACF_B64 = "dxWiJeYWsl07kkfpP9XX/H8eAH47Ex9AMGrP2GEhPXmylnaxRuyhm7Jm92wfWuRrfhC4Z2c/nyCEuHMPqnKzLfxLYEUwAoRJly4PDJ1N9tsFdoEly63K6Uu64ywKrhZQlmzJkOKxGBpmJLTFgfHzZNkEwb+7JCXzy4HimxgDZAol2iyuihI0A8X0qwqkikZjQa/w0NwGO62O9dFE+eSIh9mNlj9zBa+C2Cb0M5CX20nMpXdk7Huexg=="


def _get_acf_files():
    """Return list of ACF paths to search. Use chunk_fwd_o_2.acf (submission's ACF) + ''.
    chunk_fwd_o_0.acf etc. produce NaNs - restrict to known-good ACF."""
    local = Path("/opt/booster_pack/chunk_fwd_o_2.acf")
    if local.exists():
        return [str(local), ""]
    d = tempfile.mkdtemp(prefix="fwd_o_acf_")
    p = Path(d) / "chunk_fwd_o_2.acf"
    p.write_bytes(base64.b64decode(_ACF_B64))
    return [str(p), ""]


ALL_SHAPES = [
    (1, 64, 2, 64, 64),
    (2, 128, 4, 64, 64),
    (1, 256, 4, 64, 128),
    (1, 64, 1, 64, 64),
    (2, 512, 3, 64, 64),
    (2, 1024, 3, 64, 64),
    (3, 1024, 4, 100, 100),
    (4, 1024, 4, 128, 128),
    (2, 1536, 4, 128, 128),
    (4, 2048, 8, 64, 64),
]


def make_autotune_kernel(acf_files: list[str], effort: str):
    """Kernel with autotune enabled - matches submission logic (TF32, exp2)."""

    @helion.kernel(
        static_shapes=True,
        dot_precision="tf32",
        autotune_effort=effort,
        autotune_search_acf=acf_files,
    )
    def kernel(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        h: torch.Tensor,
        g: torch.Tensor,
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
            g_c = g[b_idx, tile_t, h_idx]
            q_c = q[b_idx, tile_t, h_idx, :]
            k_c = k[b_idx, tile_t, h_idx, :]
            v_c = v[b_idx, tile_t, h_idx, :]
            h_c = h[b_idx, c_idx, h_idx, :, :]
            o_inter = hl.dot(q_c, h_c, out_dtype=torch.float32) * torch.exp2(g_c * LOG2_E)[:, None]
            qk = hl.dot(q_c, k_c.T, out_dtype=torch.float32)
            g_diff = g_c[:, None] - g_c[None, :]
            qk = qk * torch.exp2(g_diff * LOG2_E)
            idx = hl.arange(tile_t.block_size)
            causal = (idx[:, None] >= idx[None, :])
            qk = torch.where(causal, qk, 0.0)
            o_intra = hl.dot(qk, v_c, out_dtype=torch.float32)
            out[b_idx, tile_t, h_idx, :] = ((o_inter + o_intra) * scale).to(out.dtype)
        return out

    return kernel


def config_to_submission_str(config: helion.Config, acf_var: str = "_ACF") -> str:
    """Format config for submission.py SHAPE_CONFIGS."""
    cfg = config.config
    acf = cfg.get("advanced_controls_file", "")
    if acf:
        acf_repr = acf_var
    else:
        acf_repr = "''"
    parts = [f"advanced_controls_file={acf_repr}"]
    keys = [
        "block_sizes", "indexing", "l2_groupings", "load_eviction_policies",
        "loop_orders", "num_sm_multiplier", "num_stages", "num_warps",
        "pid_type", "range_flattens", "range_multi_buffers", "range_num_stages",
        "range_unroll_factors", "range_warp_specializes",
    ]
    for k in keys:
        if k in cfg:
            v = cfg[k]
            if isinstance(v, str):
                parts.append(f'{k}="{v}"')
            else:
                parts.append(f"{k}={repr(v)}")
    return "helion.Config(" + ", ".join(parts) + ")"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--effort", default="full", choices=["quick", "full"])
    ap.add_argument("--shape", default=None, help="Comma-separated shape indices to tune (default: all)")
    args = ap.parse_args()

    shape_indices = range(len(ALL_SHAPES))
    if args.shape is not None:
        shape_indices = [int(x.strip()) for x in args.shape.split(",")]

    acf_files = _get_acf_files()
    print(f"ACF files: {acf_files}", flush=True)
    print(f"Effort: {args.effort}", flush=True)
    print(f"Shapes: {[ALL_SHAPES[i] for i in shape_indices]}", flush=True)

    results = {}
    for idx in shape_indices:
        shape = ALL_SHAPES[idx]
        B, T, H, K, V = shape
        print(f"\n[{idx}] Autotuning shape {shape}...", flush=True)
        data = generate_input(B, T, H, K, V, seed=42)
        q, k, v_new, h, g = data
        scale = K ** -0.5
        kernel_fn = make_autotune_kernel(acf_files, args.effort)
        bound = kernel_fn.bind((q, k, v_new, h, g, scale))
        try:
            best_config = bound.autotune((q, k, v_new, h, g, scale), force=True)
            results[shape] = best_config
            cfg_str = config_to_submission_str(best_config)
            print(f"  [DONE] Best config:", flush=True)
            print(f"    {cfg_str}", flush=True)
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
