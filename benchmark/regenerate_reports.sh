#!/usr/bin/env bash
# Rebuild every report, table and figure from the result records. Runs no
# benchmark and calls no model: it only re-reads benchmark/results/.
#
#   bash benchmark/regenerate_reports.sh
#
# Order matters: the figures read the reports written before them (paired.json,
# rq/report.json, ablation.json, cells), and the supplementary exports read both.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=$(mktemp -d)
run() {   # name command...
  local name=$1; shift
  if "$@" >"$LOG/$name.log" 2>&1; then echo "  ok    $name"; else echo "  FAIL  $name (log: $LOG/$name.log)"; return 1; fi
}

echo "== per-arm scoring reports"
for arm in benchmark/results/alzkb/*_*/ benchmark/results/biohopr/*_*/; do
  ls "$arm"*.jsonl >/dev/null 2>&1 || continue
  run "arm_$(basename "$arm")" poetry run python -m benchmark.evaluate_benchmark "$arm" &
done
wait

echo "== tables and statistics"
run scores        poetry run python -m benchmark.report_scores &
run rq            poetry run python -m benchmark.report_rq &   # both benchmarks: one Holm family per RQ
run paired_alzkb  poetry run python -m benchmark.report_paired_stats --preset alzkb \
                    --reference nexa-direct_gpt-5-mini -o benchmark/results/alzkb/stats/paired &
run paired_biohopr poetry run python -m benchmark.report_paired_stats --preset biohopr \
                    --reference nexa-direct_gpt-5-mini -o benchmark/results/biohopr/stats/paired &
run variability_4o poetry run python -m benchmark.evaluate_benchmark --variability \
                    benchmark/results/alzkb_variability/nexa-direct_gpt-4o-mini \
                    -o benchmark/results/alzkb_variability/stats/variability_gpt-4o-mini.txt &
run variability_5  poetry run python -m benchmark.evaluate_benchmark --variability \
                    benchmark/results/alzkb_variability/nexa-direct_gpt-5-mini \
                    benchmark/results/alzkb_variability/nexa-flex_gpt-5-mini \
                    -o benchmark/results/alzkb_variability/stats/variability_gpt-5-mini.txt &
run ablation_direct poetry run python -m benchmark.ablation_direct report &
run escargot_cypher poetry run python -m benchmark.escargot_cypher_only report &
wait

echo "== figures"
run cells         poetry run python figures/data.py
for fig in plot_alzkb_results plot_biohopr_results plot_research_questions plot_ablation plot_perturbation; do
  run "$fig" poetry run python "figures/$fig.py" &
done
wait

echo "== supplementary exports"
run supplementary poetry run python -m benchmark.export_supplementary
echo "done (logs in $LOG)"
