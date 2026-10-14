#!/usr/bin/env bash
# Run-to-run variability study: run every subset in
# benchmark/datasets/alzkb_variability/ R times through the runner, tagging the
# passes run1..runR.
#
# The model and provider come from env vars (the runner has no --model flag):
#   export BENCHMARK_PROVIDER=openai
#   export BENCHMARK_MODEL_NAME=gpt-4o-mini        # or gpt-5-mini
#
# Usage:
#   benchmark/run_variability.sh <agent-mode> <R> "<profile>"
#   benchmark/run_variability.sh direct 5 "Alzkb 121"
#   benchmark/run_variability.sh flex   5 "Alzkb 121"
#
# Resumable: each (config, dataset, run tag) has its own JSONL and the runner
# skips questions already recorded.
set -euo pipefail

AGENT_MODE="${1:-direct}"
R="${2:-5}"
PROFILE="${3:-Alzkb 121}"
# Optional 4th arg: output subdir; default is the arm directory
# alzkb_variability/nexa-<mode>_<model> (":" replaced, not allowed in Windows paths).
MODEL_SLUG="${BENCHMARK_MODEL_NAME:-unknown}"; MODEL_SLUG="${MODEL_SLUG//:/-}"
OUT_SUBDIR="${4:-alzkb_variability/nexa-${AGENT_MODE}_${MODEL_SLUG}}"
# Few-shot examples mode for direct runs ('llm', as in the published runs); unused by flex.
EXAMPLES_MODE="${EXAMPLES_MODE:-llm}"

SUBSET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/datasets/alzkb_variability"

if [[ -z "${BENCHMARK_PROVIDER:-}" || -z "${BENCHMARK_MODEL_NAME:-}" ]]; then
  echo "ERROR: set BENCHMARK_PROVIDER and BENCHMARK_MODEL_NAME first." >&2
  echo "  export BENCHMARK_PROVIDER=openai BENCHMARK_MODEL_NAME=gpt-4o-mini" >&2
  exit 1
fi
if [[ ! -d "$SUBSET_DIR" ]]; then
  echo "ERROR: $SUBSET_DIR not found; run build_variability_subset.py first." >&2
  exit 1
fi

echo "Variability run: mode=$AGENT_MODE  R=$R  model=$BENCHMARK_MODEL_NAME  profile=$PROFILE"
echo "Output dir: benchmark/results/$OUT_SUBDIR"

type_for() {
  case "$(basename "$1")" in
    MCQ_*)           echo "mcq" ;;
    True_or_False_*) echo "tf" ;;
    OpenEnded_*)     echo "open_ended" ;;
    *) echo "" ;;
  esac
}

for r in $(seq 1 "$R"); do
  tag="run${r}"
  for f in "$SUBSET_DIR"/*_sub.json; do
    btype="$(type_for "$f")"
    [[ -z "$btype" ]] && { echo "  skip (unknown type): $f"; continue; }
    echo ">>> [$tag] $btype  $(basename "$f")"
    extra=()
    [[ "$AGENT_MODE" == "direct" ]] && extra=(--examples-mode "$EXAMPLES_MODE")
    poetry run python -m benchmark.benchmark \
      --type "$btype" --input "$f" \
      --agent-mode "$AGENT_MODE" --profile "$PROFILE" --run-tag "$tag" \
      --out-subdir "$OUT_SUBDIR" "${extra[@]}"
  done
done

echo "Done. Score with:"
echo "  poetry run python -m benchmark.evaluate_benchmark benchmark/results/$OUT_SUBDIR --variability"
