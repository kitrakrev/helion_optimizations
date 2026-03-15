#!/usr/bin/env python3
"""
Local Nsight Compute profiling for gated_deltanet_chunk_fwd_h.

Run with Nsight Compute (ncu) to get detailed GPU metrics:
  ncu -o fwd_h_profile --set basic python profile_nsight.py [benchmark_idx]

  # Full metrics (slower):
  ncu -o fwd_h_profile --set full python profile_nsight.py

Output: gated_deltanet_chunk_fwd_h_py_profile.ncu-rep (when run via eval.py profile-nsight)
"""

import sys
from pathlib import Path

_PROBLEM_DIR = Path(__file__).resolve().parent
_HELION_DIR = _PROBLEM_DIR.parent
sys.path.insert(0, str(_HELION_DIR))
sys.path.insert(0, str(_PROBLEM_DIR))
import os

os.chdir(_PROBLEM_DIR)

import yaml
import torch
from utils import set_seed


def main():
    set_seed(42)

    task_path = _PROBLEM_DIR / "task.yml"
    task = yaml.safe_load(task_path.read_text())
    benchmarks = task.get("benchmarks", [])

    bench_idx = 0
    if len(sys.argv) > 1:
        bench_idx = int(sys.argv[1])

    if bench_idx >= len(benchmarks):
        print(f"Benchmark index {bench_idx} out of range (0..{len(benchmarks)-1})", file=sys.stderr)
        sys.exit(1)

    bench = benchmarks[bench_idx]
    print(f"Profiling benchmark {bench_idx}: {bench}", file=sys.stderr)

    from submission import custom_kernel
    from reference import generate_input

    data = generate_input(**bench)
    torch.cuda.synchronize()

    for _ in range(5):
        _ = custom_kernel(data)
    torch.cuda.synchronize()

    _ = custom_kernel(data)
    torch.cuda.synchronize()

    print("Done. ncu will have captured the kernel.", file=sys.stderr)


if __name__ == "__main__":
    main()
