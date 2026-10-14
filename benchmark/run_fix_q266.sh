#!/usr/bin/env bash
# Run AlzKB MCQ 1-hop entry 266 ("Which of the following diseases presents the
# symptom Paraparesis, Spastic?") for every arm, then merge the records
# (see benchmark/fix_mcq1hop_q266.py). Each command reproduces its arm's
# original configuration, so the new record is comparable with the other 497.
#
#   bash benchmark/run_fix_q266.sh openai     # the 7 OpenAI arms
#   bash benchmark/run_fix_q266.sh qwen       # the qwen arms (NEXA Direct, Flex, vanilla, GraphCypherQAChain)
#
# Safe to re-run: the baselines resume their own file; the NEXA runner writes to
# benchmark/results/alzkb_q266/<arm>/, and only `merge` changes the original
# result files, with backups.
set -euo pipefail
cd "$(dirname "$0")/.."

R=benchmark/results/alzkb
Q=benchmark/datasets/alzkb_q266/MCQ_1hop.json
one() { ls $1 | head -n 1; }     # the single file an arm glob points at

nexa() {   # arm provider model agent_mode [extra env...]
  local arm=$1 provider=$2 model=$3 mode=$4; shift 4
  local extra=(--examples-mode llm --generation-mode one_shot_fallback)
  [ "$mode" = flex ] && extra=()
  echo "== $arm"
  env BENCHMARK_PROVIDER="$provider" BENCHMARK_MODEL_NAME="$model" "$@" \
    poetry run python -m benchmark.benchmark --type mcq --input "$Q" \
      --agent-mode "$mode" --profile "Alzkb 121" "${extra[@]}" \
      --out-subdir "alzkb_q266/$arm"
}

baseline() {   # arm script provider model resume-glob [extra args...]
  local arm=$1 script=$2 provider=$3 model=$4 glob=$5; shift 5
  echo "== $arm"
  poetry run python "benchmark/baselines/$script" --provider "$provider" --model "$model" \
    --datasets MCQ_1hop.json --resume "$(one "$R/$arm/$glob")" "$@"
}

case "${1:-}" in
  openai)
    baseline vanilla_gpt-4o-mini     baseline_vanilla.py     openai gpt-4o-mini 'baseline_vanilla_gpt-4o-mini_2*.jsonl'
    baseline vanilla_gpt-5-mini      baseline_vanilla.py     openai gpt-5-mini  'baseline_vanilla_gpt-5-mini_run1_*.jsonl'
    baseline graphcypher_gpt-4o-mini baseline_graphcypher.py openai gpt-4o-mini 'baseline_graphcypher_*_topk10000_*.jsonl' --top-k 10000
    baseline graphcypher_gpt-5-mini  baseline_graphcypher.py openai gpt-5-mini  'baseline_graphcypher_*_run1_*.jsonl' --top-k 10000

    nexa nexa-direct_gpt-4o-mini openai gpt-4o-mini direct
    nexa nexa-direct_gpt-5-mini  openai gpt-5-mini  direct
    nexa nexa-flex_gpt-5-mini    openai gpt-5-mini  flex
    ;;
  qwen)
    # Same flags as the original qwen runs (read from their .meta.json).
    nexa nexa-direct_qwen3.8-27b ollama qwen3.8:27b-q8_0 direct NEXA_LLM_MAX_OUTPUT_TOKENS=32768
    nexa nexa-flex_qwen3.8-27b   ollama qwen3.8:27b-q8_0 flex \
      NEXA_LLM_MAX_OUTPUT_TOKENS=8192 NEXA_FLEX_ANSWER_RESCUE=true
    baseline vanilla_qwen3.8-27b baseline_vanilla.py ollama qwen3.8:27b-q8_0 'baseline_vanilla_qwen3.8_27b-q8_0_2*.jsonl'
    # This arm's file name does not include the model name.
    baseline graphcypher_qwen3.8-27b baseline_graphcypher.py ollama qwen3.8:27b-q8_0 'baseline_graphcypher_Alzkb_121_2*.jsonl' --top-k 10000
    ;;
  *)
    echo "usage: bash benchmark/run_fix_q266.sh {openai|qwen}" >&2
    exit 2
    ;;
esac

poetry run python -m benchmark.fix_mcq1hop_q266 merge
