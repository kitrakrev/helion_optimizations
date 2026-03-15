#!/usr/bin/env python3
"""Autotune gated_deltanet_chunk_fwd_h and print best configs for submission.py.

Usage:
  cd reference-kernels/problems/helion
  python gated_deltanet_chunk_fwd_h_py/autotune.py [--effort quick|full] [--shape 0,1,2,...]

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

# ACF: use /opt/booster_pack if available, else embedded
_ACF_B64 = "dxWiJeZ70zxa8yaIXrS2nR5/YR9acn4hUQuuuQBAtk1NaYlOuRNeZE2ZCJPgpRuUgTkBm9Xup9NUyKFZuxpQHy4NGUpw0lKHWcV8hdhPD4bwug64u+dHVGJ0O1wdbs4aUDmcxnHy3RxJxoEtFfIzKmYi8w3kKjm8Y2AZMLMAEH4cJVgYTO6E07mI32mTktUpVB58dOGBr4ZO/U3hTkwLNIfaQR/MHiD+MNq1rpg1mdzFMNLBwhL73AKIz7nG3Y5Mqg9dt968yZzFOuWzYJmfpzsRy3BnDkcQa9YkYnG7pSKlsgfJcj0XTD4+cLp24kIcOdZnfZGKrYr0qO/0M8lBaL3bqTVAsaBEOfohEQBZUHQsTq6sSQ9qI/aV3lCFkbK0ipM6PmaEku/WtNU2batXFl9jNgp61oUdKSz7zeengbw+cVypqN1g8UI3InvHbFEkvdnu8FUiMvK1A+H6La9r1w3Mz6LmxEKemrxphszkpuNQRaN48swQ2/QB5nNzH0SbmaR03UKyM1DQDd9r2YAzIj772mC2j+J+fO9gSEqTclYE0+LsFb0z/oT4p5KGmR7uPAFTJw4kVwb7oPz3SKvIkPehXIROtufCad+Fix/mpuFevP+hhtp8qNkRkKhGBiygvSCcSCQPJRphf+VcYxuhCZs3QE1neTVnOqA/R6njcVwP5sq/FhwkAmfCbq5VkrVseUEyS7fdQGCXHl1mIR2xgFWr9+jwXw4VNTgMED7f8eLOknaM8/M+3dnYn1xhPfujAZt4ROgFiMNCnM9zDZmYjoP8R6JQnViTouHpfDnHEo3/u4tZKtCHvfoBRKGOSDurka18H6DUDACOlySq7yo9hOvHgQ3Ed5W9pRGPNDExivqcw8lyUVo2TMXMvI4Ld0wvpzuQsbxwQ8+G3ZCI7EECfqDpTSik6z9QkGP5cjyAHgUli/5eunOGldC7iDMRsLxIJmkBchh1xkCmO5pYiqOjeSnYETAvIGWdTyTXksi8+eoPlV6yhQgqhTdCGQM3Oa52Qyvnh4vQYEur7JeiqlMvFmI="


def _get_acf_files():
    """Return list of ACF paths to search: [acf_path, ''] to try with and without ACF."""
    local = Path("/opt/booster_pack/chunk_fwd_h_0.acf")
    if local.exists():
        acf_path = str(local)
    else:
        d = tempfile.mkdtemp(prefix="fwd_h_acf_")
        p = Path(d) / "chunk_fwd_h_0.acf"
        p.write_bytes(base64.b64decode(_ACF_B64))
        acf_path = str(p)
    return [acf_path, ""]


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
        k: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        g: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H, K = k.shape
        V = u.shape[-1]
        C = 64
        K = hl.specialize(K)
        V = hl.specialize(V)
        NT = (T + C - 1) // C
        h_out = torch.empty(B, NT, H, K, V, dtype=k.dtype, device=k.device)
        v_out = torch.empty_like(u)
        BH = B * H
        for flat, tv in hl.tile([BH, V], block_size=[1, None]):
            b_idx = flat.begin // H
            h_idx = flat.begin % H
            state = hl.zeros([K, tv], dtype=torch.float32)
            for tc in hl.tile(T, block_size=C):
                chunk_idx = tc.begin // C
                t_end = tc.begin + C - 1
                h_out[b_idx, chunk_idx, h_idx, :, tv] = state.to(k.dtype)
                proj = hl.dot(w[b_idx, tc, h_idx, :], state, out_dtype=torch.float32)
                diff = u[b_idx, tc, h_idx, tv].to(torch.float32) - proj
                v_out[b_idx, tc, h_idx, tv] = diff.to(u.dtype)
                g_end = g[b_idx, t_end, h_idx]
                g_t = g[b_idx, tc, h_idx]
                alpha = torch.exp2((g_end - g_t) * LOG2_E)
                k_adj = k[b_idx, tc, h_idx, :] * alpha[:, None]
                state = state * torch.exp2(g_end * LOG2_E)
                state = state + hl.dot(k_adj.T, diff, out_dtype=torch.float32)
        return h_out, v_out

    return kernel


def config_to_submission_str(config: helion.Config, acf_var: str = "_ACF") -> str:
    """Format config for submission.py SHAPE_CONFIGS. Use _ACF for ACF path.
    Omits static_ranges if kernel has 0 range loops (avoids InvalidConfig)."""
    cfg = config.config
    acf = cfg.get("advanced_controls_file", "")
    if acf:
        acf_repr = acf_var
    else:
        acf_repr = "''"
    parts = [f"advanced_controls_file={acf_repr}"]
    keys = ["block_sizes", "indexing", "l2_groupings", "load_eviction_policies",
            "loop_orders", "num_sm_multiplier", "num_stages", "num_warps",
            "pid_type", "range_flattens", "range_multi_buffers", "range_num_stages",
            "range_unroll_factors", "range_warp_specializes"]
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
        k, w, u, g = data
        kernel_fn = make_autotune_kernel(acf_files, args.effort)
        bound = kernel_fn.bind((k, w, u, g))
        try:
            best_config = bound.autotune((k, w, u, g), force=True)
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
