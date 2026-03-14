#!/bin/bash
# Submit all Helion problems to the leaderboard.
# Run from: reference-kernels/problems/helion/
# Usage: ./submit_all_leaderboard.sh

set -e
cd "$(dirname "$0")"

PROBLEMS=(
  "causal_conv1d_py"
  "fp8_quant_py"
  "gated_deltanet_recompute_w_u_py"
  "gated_deltanet_chunk_fwd_o_py"
  "gated_deltanet_chunk_fwd_h_py"
)

echo "Submitting ${#PROBLEMS[@]} problems to leaderboard..."
for p in "${PROBLEMS[@]}"; do
  echo ""
  echo "=== Submitting $p ==="
  popcorn submit "$p/submission.py" --mode leaderboard --no-tui || echo "  (submit failed or skipped)"
done
echo ""
echo "Done."
