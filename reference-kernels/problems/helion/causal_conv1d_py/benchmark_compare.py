#!/usr/bin/env python3
"""Compare submission variants: loop, reduce, tileir."""
import sys
import os
from pathlib import Path

# Add parent for utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent)

import yaml
import torch
from utils import set_seed

def run_benchmark(module_name, num_runs=100):
    """Run benchmark for a submission variant. module_name='' = baseline (no swap)."""
    for m in list(sys.modules.keys()):
        if 'submission' in m:
            del sys.modules[m]
    
    submission_py = Path('submission.py')
    if module_name:
        variant_py = Path(f'submission_{module_name}.py')
        if not variant_py.exists():
            return None, f"File not found: {variant_py}"
        backup = submission_py.read_text()
        submission_py.write_text(variant_py.read_text())
    
    try:
        from submission import custom_kernel
        from reference import generate_input
        
        task = yaml.safe_load(Path('task.yml').read_text())
        benchmarks = task.get('benchmarks', [])
        
        results = []
        for bench in benchmarks:
            set_seed(42)
            data = generate_input(**bench)
            torch.cuda.synchronize()
            # Warmup
            for _ in range(20):
                custom_kernel(data)
            torch.cuda.synchronize()
            # Timed runs
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(num_runs):
                custom_kernel(data)
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / num_runs
            results.append((bench, ms))
        return results, None
    except Exception as e:
        return None, str(e)
    finally:
        if module_name:
            submission_py.write_text(backup)

def run_test(module_name):
    """Run correctness test. module_name='' = baseline."""
    for m in list(sys.modules.keys()):
        if 'submission' in m:
            del sys.modules[m]
    
    submission_py = Path('submission.py')
    if module_name:
        variant_py = Path(f'submission_{module_name}.py')
        if not variant_py.exists():
            return False, f"File not found"
        backup = submission_py.read_text()
        submission_py.write_text(variant_py.read_text())
    
    try:
        from submission import custom_kernel
        from reference import check_implementation, generate_input
        
        task = yaml.safe_load(Path('task.yml').read_text())
        tests = task.get('tests', [])
        for t in tests:
            data = generate_input(**t)
            if not check_implementation(custom_kernel, data):
                return False, f"Failed test {t}"
        return True, None
    except Exception as e:
        return False, str(e)
    finally:
        if module_name:
            submission_py.write_text(backup)

def main():
    set_seed(42)
    
    variants = ['reduce', 'tileir']
    # Baseline is current submission.py (loop)
    
    print("=" * 60)
    print("CORRECTNESS TESTS")
    print("=" * 60)
    
    # Test baseline
    print("\nBaseline (submission.py - loop):")
    ok, err = run_test('')
    if ok:
        print("  PASS")
    else:
        print(f"  FAIL: {err}")
    
    for v in variants:
        print(f"\nVariant {v}:")
        ok, err = run_test(v)
        if ok:
            print("  PASS")
        else:
            print(f"  FAIL: {err}")
    
    print("\n" + "=" * 60)
    print("BENCHMARKS (mean ms)")
    print("=" * 60)
    
    # Baseline
    print("\nBaseline (submission.py - loop):")
    results, err = run_benchmark('')
    if err:
        print(f"  ERROR: {err}")
        baseline_results = {}
    elif results:
        for bench, ms in results:
            print(f"  {bench}: {ms:.4f} ms")
        baseline_results = dict((str(b), ms) for b, ms in results)
    else:
        baseline_results = {}
    
    # Variants
    for v in variants:
        print(f"\nVariant {v}:")
        results, err = run_benchmark(v)
        if err:
            print(f"  ERROR: {err}")
            continue
        for bench, ms in results:
            bstr = str(bench)
            base = baseline_results.get(bstr, ms)
            diff = ((ms - base) / base * 100) if base else 0
            print(f"  {bench}: {ms:.4f} ms ({diff:+.1f}% vs baseline)")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
