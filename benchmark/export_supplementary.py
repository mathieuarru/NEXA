"""Raw exports for the Supplementary: ablation, perturbations, paired comparisons.

Reads what the reports and figure scripts compute, so every number matches a
figure or a report table:

  component_ablation.csv     the nine ablations of NEXA Direct (GPT-4o-mini) on
                             AlzKB and BioHopR. Rows at level "figure" are the
                             points of the ablation figure (BioHopR: Jaccard;
                             AlzKB: pooled primary metric, Holm over the nine
                             arms); rows at level "subset" are AlzKB per question
                             set (Holm over the 54 comparisons); rows at level
                             "subgroup" read the two example arms on the AlzKB
                             questions that do / do not cite an example literal
                             (descriptive, raw p only).
                             From results/*_ablation/stats/ablation.json.
  entity_perturbations.csv   the three value-correction ablations on perturbed
                             entity names, per benchmark x perturbation, in both
                             metrics of the perturbation figures: "score"
                             (AlzKB pooled primary metric, BioHopR Jaccard) and
                             "exact" (exact answer). Holm over the 12
                             comparisons of each benchmark and metric. Computed
                             by figures/plot_perturbation.py's own loaders.
  paired_comparisons.csv     every RQ1-RQ3 comparison (and the RQ1 secondary one
                             against the LLM alone), original question sets and
                             the disambiguated True/False 2-hop set, with the
                             Holm family each was corrected in.
                             From results/{alzkb,biohopr}/stats/rq/report.json.
  experimental_details.txt   definitions, sampling, perturbation rules, metrics,
                             tests, families, seeds, software, run provenance.

    poetry run python -m benchmark.export_supplementary   # -> benchmark/results/supplementary/

Run it after benchmark.report_rq and benchmark.ablation_direct report (the
regenerate script does).
"""

from __future__ import annotations

import csv
import glob
import importlib.metadata as md
import json
import os
import platform
import sys
from collections import Counter
from typing import Any, Dict, List

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_REPO, os.path.join(_REPO, "figures")):
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmark.ablation_direct import (ARMS, DATASETS, EXPERIMENTS, MODEL, PINNED_ENV,  # noqa: E402
                                       arm_dir, load_experiment_arm, load_json, usage_summary)
from benchmark.paired_stats import N_BOOT  # noqa: E402
import plot_ablation  # noqa: E402
import plot_perturbation  # noqa: E402

RESULTS = os.path.join(_REPO, "benchmark", "results")
OUT = os.path.join(RESULTS, "supplementary")
BOOT_SEED = 20260921          # paired_stats.paired_bootstrap_diff / mean_ci_bca default seed

# Ablation figure labels and order, taken from the figure script.
FIGURE10 = [(arm, label, group) for group, rows in plot_ablation.GROUPS for arm, label in rows]
SET_LABEL = {"True_or_False_1hop.json": "TF 1-hop", "True_or_False_2hop.json": "TF 2-hop",
             "MCQ_1hop.json": "MCQ 1-hop", "MCQ_2hop.json": "MCQ 2-hop",
             "OpenEnded_1hop.json": "OE 1-hop", "OpenEnded_2hop.json": "OE 2-hop",
             "BioHopR_2hop.json": "BioHopR 2-hop",
             "True_or_False_2hop_atleastone.json": "TF 2-hop (disambiguated)"}

# What each arm switches off and what it leaves untouched (benchmark/ablation_direct.py ARMS,
# nexa/config/params.py NEXA_ABLATE_*, nexa/knowledge_graph/graph_query.py, query_correction.py).
DEFINITIONS = {
    "no-examples": ("Every Direct prompt is zero-shot: the few-shot block of the label-selection, "
                    "relationship-selection, Cypher-generation and query-correction prompts is empty "
                    "(NEXA_ABLATE_FEWSHOT_EXAMPLES).",
                    "Everything else, including the identifying-property table given to the generation "
                    "and correction prompts, the one-shot-then-fallback strategy and the whole correction loop."),
    "generic-examples": ("The graph-generated examples are replaced by the generic examples of "
                         "nexa/models/prompts.py: the same query shapes (single hop, chain, union, "
                         "intersection, list filter) written with <LABEL>/<REL_TYPE> placeholders, "
                         "nothing taken from the graph (examples_mode = hardcoded).",
                         "Number and position of the example blocks, every other prompt and component."),
    "bare": ("The whole recovery path: one one-shot query, executed once, its result final. No staged "
             "fallback, no empty-result correction (value matching, LLM selection, rewrite), no repair "
             "of Neo4j errors (generation_mode = one_shot, NEXA_ABLATE_EMPTY_RESULT_LOOP, "
             "NEXA_ABLATE_SYNTAX_CORRECTION).",
             "The one-shot generation prompt with its examples and identifying properties, answer synthesis."),
    "no-empty-loop": ("The empty-result correction loop: an empty result after the staged fallback is "
                      "accepted as final, without extracting the query's literals, matching them "
                      "against the database, selecting a candidate or rewriting the query "
                      "(NEXA_ABLATE_EMPTY_RESULT_LOOP).",
                      "One-shot first, the staged fallback when the one-shot query returns nothing, "
                      "and the repair of Neo4j errors."),
    "one-shot": ("The staged (decomposed) fallback: when the one-shot query returns nothing, the "
                 "correction loop repairs that query instead of regenerating it through label "
                 "selection, relationship selection and staged Cypher generation "
                 "(generation_mode = one_shot).",
                 "The whole correction loop (value matching, LLM selection, rewrite, error repair)."),
    "no-db-matching": ("Database value matching inside the correction loop: the loop still runs and "
                       "the LLM still rewrites the query, but without candidate values retrieved from "
                       "the graph for the query's literals (NEXA_ABLATE_DB_MATCHING).",
                       "One-shot first, staged fallback, the rewrite step and error repair."),
    "no-llm-selection": ("LLM contextual selection: candidate values are still retrieved from the "
                         "graph, but the top-ranked one is used as is; no LLM choice among the "
                         "candidates and no LLM-proposed synonym (NEXA_ABLATE_LLM_SELECTION).",
                         "One-shot first, staged fallback, database matching, the rewrite and error repair."),
    "no-syntax": ("Syntactic correction: Neo4j errors and notifications are not sent to the "
                  "dedicated repair prompt; a failing query is handled like an empty one "
                  "(NEXA_ABLATE_SYNTAX_CORRECTION).",
                  "One-shot first, staged fallback and the empty-result correction loop."),
    "decomposed": ("The one-shot-first strategy: every question takes the staged path (label "
                   "selection, relationship selection, Cypher generation) from the start, then the "
                   "correction loop (generation_mode = decomposed).",
                   "Every prompt and the whole correction loop."),
}


def pct(x):
    """Fraction as a percentage rounded to 4 decimals; None stays None."""
    return None if x is None else round(100 * x, 4)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write rows to CSV; the columns are every key, in first-seen order."""
    keys: List[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# 1. Component ablation
# ---------------------------------------------------------------------------

def component_ablation() -> List[Dict[str, Any]]:
    """Rows of component_ablation.csv, from each experiment's stats/ablation.json."""
    rows = []
    for exp_key, bench, metric in (("biohopr", "BioHopR", "Jaccard index"),
                                   ("alzkb", "AlzKB", "pooled primary metric")):
        exp = EXPERIMENTS[exp_key]
        rep = load_json(os.path.join(RESULTS, exp.name, "stats", "ablation.json"))
        full_usage = usage_summary(load_experiment_arm(exp.name, arm_dir("full")))
        for arm, label, group in FIGURE10:
            a, spec = rep["arms"][arm], ARMS[arm]
            base = {"benchmark": bench, "variant": label, "group": group, "arm": arm,
                    "generation_mode": spec.generation_mode, "examples_mode": spec.examples_mode,
                    "flags": " ".join(spec.flags) or "-",
                    "tokens_per_q_variant": round(a["usage"]["tokens_per_q"], 1),
                    "tokens_per_q_full": round(full_usage["tokens_per_q"], 1),
                    "latency_s_per_q_variant": round(a["usage"]["latency_s_per_q"], 2),
                    "latency_s_per_q_full": round(full_usage["latency_s_per_q"], 2)}
            if exp_key == "biohopr":
                cells = [("figure", "BioHopR 2-hop", metric, a["sets"]["BioHopR_2hop.json"], rep["holm_family"])]
            else:
                cells = [("figure", "six subsets pooled", metric, a["pooled"], rep["pooled_holm_family"])]
                # Example arms on questions citing an example literal or not (raw p only, no Holm).
                for sub, label in (("cites_example_entity", "pooled, questions citing an example entity"),
                                   ("no_example_entity", "pooled, other questions")):
                    if a["pooled"].get(sub):
                        cells.append(("subgroup", label, metric, {"all": a["pooled"][sub]}, None))
                cells += [("subset", SET_LABEL[s], v["metric"], v, rep["holm_family"]) for s, v in a["sets"].items()]
            for level, subset, m, entry, fam in cells:
                c, t = entry["all"], entry.get("triggered") or {}
                rows.append({**{k: base[k] for k in ("benchmark", "variant", "group", "arm")},
                             "level": level, "subset": subset, "metric": m,
                             "n": c["n"], "score_full": pct(c["b"]), "score_variant": pct(c["a"]),
                             "delta_pp": pct(c["delta"]), "ci95_lo_pp": pct(c["ci"][0]),
                             "ci95_hi_pp": pct(c["ci"][1]), "test": c["test"], "p_raw": c["p"],
                             "p_raw_at_floor": c["test"] == "paired_bootstrap_bca" and c["p"] <= 1 / N_BOOT,
                             "p_holm": c.get("p_holm") if fam else None, "holm_family_size": fam,
                             "questions_gained": c["gained"], "questions_lost": c["lost"],
                             "n_triggered": t.get("n"), "delta_triggered_pp": pct(t.get("delta")),
                             **{k: base[k] for k in ("generation_mode", "examples_mode", "flags",
                                                     "tokens_per_q_variant", "tokens_per_q_full",
                                                     "latency_s_per_q_variant", "latency_s_per_q_full")}})
    return rows


# ---------------------------------------------------------------------------
# 2. Entity-name perturbations (the loaders of the perturbation figures)
# ---------------------------------------------------------------------------

def entity_perturbations() -> List[Dict[str, Any]]:
    """Rows of entity_perturbations.csv, from the perturbation figure's loaders."""
    rows = []
    for key, title, graded_name in plot_perturbation.PANELS:
        bench = "AlzKB" if key.startswith("alzkb") else "BioHopR"
        for metric, loader, metric_name in (
                ("score", plot_perturbation.graded_data, graded_name),
                ("exact", plot_perturbation.accuracy_data, "exact answer (MCQ option or exact set match)")):
            data = loader(key)
            for arm, label, _ in plot_perturbation.ABLATED:
                for pert, pert_label in plot_perturbation.PERTURBATIONS:
                    d = data["diff"].get(arm, {}).get(pert)
                    if not d:
                        continue
                    full_cell, arm_cell = data["cells"]["full"][pert], data["cells"][arm][pert]
                    rows.append({"benchmark": bench, "kg": "AlzKB v1.2.1" if bench == "AlzKB" else "PrimeKG v2.1",
                                 "metric": metric, "metric_definition": metric_name,
                                 "variant": label, "arm": arm, "perturbation": pert_label,
                                 "n": d["n"], "score_full": round(full_cell[1], 4),
                                 "score_variant": round(arm_cell[1], 4),
                                 "delta_pp": pct(d["delta"]), "ci95_lo_pp": pct(d["ci"][0]),
                                 "ci95_hi_pp": pct(d["ci"][1]),
                                 "test": d.get("test", "paired_bootstrap_bca"), "p_raw": d["p"],
                                 "p_holm": d["p_holm"], "holm_family_size": 12,
                                 "questions_gained": d.get("gained"), "questions_lost": d.get("lost")})
    return rows


# ---------------------------------------------------------------------------
# 3. Paired comparisons RQ1-RQ3
# ---------------------------------------------------------------------------

SYSTEM = {"nexa-direct": "NEXA Direct", "nexa-flex": "NEXA Flex",
          "graphcypher": "GraphCypherQAChain", "vanilla": "LLM only"}
MODEL_LABEL = {"gpt-4o-mini": "GPT-4o-mini", "gpt-5-mini": "GPT-5-mini", "qwen3.8-27b": "Qwen3.8-27B"}


def system_name(arm: str) -> str:
    system, model = arm.split("_", 1)
    return f"{SYSTEM[system]} {MODEL_LABEL[model]}"


def paired_comparisons() -> List[Dict[str, Any]]:
    """Rows of paired_comparisons.csv, from each benchmark's stats/rq/report.json."""
    rows = []
    reports = {b: load_json(os.path.join(RESULTS, b, "stats", "rq", "report.json")) for b in ("alzkb", "biohopr")}
    for key in ("RQ1", "RQ2", "RQ3", "RQ1_secondary"):
        for variant, suffix in (("original", ""), ("disambiguated", "_sensitivity")):
            for b, rep in reports.items():
                q = rep.get(key + suffix)
                if not q:
                    continue
                if variant == "original":
                    fam = q["family"]
                    family = f"{key} (AlzKB + BioHopR)"
                    size = fam["size"]
                else:
                    family, size = "Negation sensitivity (AlzKB only)", rep["sensitivity_family"]["size"]
                for r in q["rows"]:
                    base = {"rq": key, "question_variant": variant, "holm_family": family,
                            "holm_family_size": size, "benchmark": "AlzKB" if b == "alzkb" else "BioHopR",
                            "question_subset": SET_LABEL[r["dataset"]], "metric": r["metric"],
                            "backbone": r["backbone"], "mode": r["label"],
                            "system_a": system_name(r["a"]), "system_b": system_name(r["b"]),
                            "delta_definition": f"{system_name(r['a'])} - {system_name(r['b'])}",
                            "status": r["status"]}
                    if r["status"] != "ok":
                        rows.append(base)
                        continue
                    rows.append({**base, "n": r["n"], "score_a": round(r["score_a"], 4),
                                 "score_b": round(r["score_b"], 4), "delta_pp": round(r["delta"], 4),
                                 "ci95_lo_pp": round(r["ci95"][0], 4), "ci95_hi_pp": round(r["ci95"][1], 4),
                                 "test": r["test"], "discordant_a_only": r.get("b_only_a"),
                                 "discordant_b_only": r.get("c_only_b"), "p_raw": r["p"],
                                 "p_raw_at_floor": bool(r.get("p_floor") and r["p"] <= r["p_floor"] + 1e-15),
                                 "p_holm": r.get("p_holm"),
                                 "partial_run": "; ".join(f"{a} ({r['n']} of {r['set_size']})" for a in r.get("partial") or []) or ""})
    return rows


# ---------------------------------------------------------------------------
# 4. Experimental details
# ---------------------------------------------------------------------------

def provenance(arm_glob: str) -> Dict[str, Any]:
    """Answer count, date range (record timestamps) and commits (.meta.json, last session only) of an arm."""
    commits, stamps, untimed, n = Counter(), [], 0, 0
    for path in glob.glob(arm_glob + ".meta.json"):
        commits[load_json(path).get("git_commit", "?")] += 1
    for path in glob.glob(arm_glob):
        if path.endswith(".meta.json"):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                n += 1
                t = json.loads(line).get("timestamp")
                if t:
                    stamps.append(t[:10])
                else:
                    untimed += 1
    return {"answers": n, "from": min(stamps) if stamps else None, "to": max(stamps) if stamps else None,
            "untimed": untimed, "commits": dict(commits)}


def fmt_provenance(pv: Dict[str, Any]) -> str:
    dates = f"{pv['from']} to {pv['to']}" if pv["from"] else "no timestamps"
    extra = f" ({pv['untimed']} answers without timestamp)" if pv["untimed"] and pv["from"] else ""
    commits = f"; last-session commits {pv['commits']}" if pv["commits"] else "; no run metadata"
    return f"{pv['answers']:5d} answers, {dates}{extra}{commits}"


def details() -> str:
    """Text of experimental_details.txt."""
    L: List[str] = []
    add = L.append
    add("NEXA: experimental details for the Supplementary (generated by benchmark/export_supplementary.py)")
    add("=" * 100)

    add("\n1. COMPONENT ABLATION: what each variant switches off, and what it leaves unchanged")
    add("   Reference ('full'): NEXA Direct as published. One-shot Cypher generation first; only if its result")
    add("   is empty, staged regeneration (label selection, relationship selection, Cypher generation) followed")
    add("   by the correction loop: a Neo4j error -> dedicated syntax repair (<= 3 rounds); an empty result ->")
    add("   the query's literals are matched against the values stored in the graph, the LLM picks among the")
    add("   top candidates or proposes a synonym, then rewrites the query (<= 3 rounds); still empty -> accepted")
    add("   as factual absence. Graph-generated few-shot examples (examples_mode = llm) in every Direct prompt.")
    for arm, label, group in FIGURE10:
        off, kept = DEFINITIONS[arm]
        spec = ARMS[arm]
        add(f"\n   {label}  [arm '{arm}'; group: {group}]")
        add(f"     switched off : {off}")
        add(f"     unchanged    : {kept}")
        add(f"     run settings : generation_mode={spec.generation_mode}, examples_mode={spec.examples_mode}, "
            f"flags={' '.join(spec.flags) or 'none'}")
    add("\n   Every component except the examples and the generation strategy acts only on questions whose")
    add("   one-shot query returned no rows or failed ('triggered' questions); on the others every arm follows")
    add("   the same path.")

    add("\n2. SAMPLES")
    for key in ("alzkb", "biohopr"):
        m = load_json(os.path.join(DATASETS, EXPERIMENTS[key].name, "manifest.json"))
        rep = m.get("representativeness") or {}
        add(f"   {EXPERIMENTS[key].description}: n = {m['n']} of {m['population_n']}, seed {m['seed']}, strata: {m['strata']}.")
        if m.get("per_set"):
            add("     per subset: " + "; ".join(f"{SET_LABEL[s]} {v['sample']}/{v['population']}" for s, v in m["per_set"].items()))
        if rep:
            add(f"     representativeness check: the full-set score of {rep.get('inside')} of {rep.get('checked')} complete "
                "configurations falls inside the 95% interval of the same configuration on the sample.")
    add("   Allocation: proportional to stratum size, largest remainder; within a stratum, a permutation drawn by a")
    add("   generator seeded from (seed, stratum), so adding or resizing a stratum does not change the others.")

    add("\n3. ENTITY-NAME PERTURBATIONS")
    for key in ("alzkb-perturbation", "biohopr-perturbation"):
        m = load_json(os.path.join(DATASETS, EXPERIMENTS[key].name, "manifest.json"))
        add(f"   {EXPERIMENTS[key].description}")
        add(f"     eligibility : {m['eligibility']} ({m['n_eligible']} eligible; not eligible: {m['not_eligible']})")
        add(f"     base sample : {m['n_base']} questions, seed {m['seed']}, strata: {m['strata']}"
            + (f"; start types {m['base_by_start_type']}" if m.get("base_by_start_type") else ""))
        for k, v in m["perturbations"].items():
            add(f"     {k:8s}: {v}")
        add(f"     dropped     : {m['dropped'] or 'none'} (a case or typo variant that is itself the exact name of "
            "another node would change the question, and so its gold)")
        add(f"     items       : {m['counts']} (gold answers unchanged)")
    add("   Why fewer than 150: the typo needs a name of 5+ characters (and no collision); the full name exists")
    add("   only for genes (AlzKB: Gene.commonName; BioHopR: NCBI full name through AlzKB, a single match required).")
    add("   Arms: complete NEXA Direct and three value-correction ablations (no-empty-loop, no-db-matching,")
    add("   no-llm-selection), GPT-4o-mini. Metrics: 'score' = AlzKB pooled primary metric (MCQ accuracy, open-ended")
    add("   Jaccard), BioHopR Jaccard; 'exact' = exact answer (MCQ option, exact set match).")

    add("\n4. METRICS AND AGGREGATION")
    add("   Primary metric per question: accuracy (0/1) for True/False and MCQ; Jaccard index between the predicted")
    add("   and gold entity sets for open-ended questions and BioHopR. Entity matching: NFKC, curly quotes and")
    add("   dashes folded, invisible characters removed, whitespace collapsed, trailing period removed, casefold.")
    add("   AlzKB pooled primary metric (ablation figure): each of the 1,000 sampled questions keeps the primary")
    add("   metric of its subset; the pooled score is their unweighted mean over all questions (subsets therefore")
    add("   weigh by their sample size), and the difference is tested with the paired bootstrap below.")

    add("\n5. STATISTICAL TESTS")
    add("   Binary outcomes (accuracy): exact two-sided McNemar test on the discordant questions (binomial test of")
    add("   min(b, c) of b + c at p = 0.5, scipy.stats); 95% interval of the paired difference by Newcombe's")
    add("   method 10 (Wilson intervals combined with the observed phi correlation).")
    add(f"   Graded outcomes (Jaccard, pooled metric): 95% BCa bootstrap interval of the mean paired difference,")
    add(f"   resampling questions with their pair of answers ({N_BOOT:,} resamples; acceleration from the jackknife);")
    add(f"   two-sided sign-flip permutation test on the per-question differences ({N_BOOT:,} sign assignments);")
    add(f"   a p at the resolution limit is reported as p < {1 / N_BOOT:.0e}. Generator: numpy default_rng, seed")
    add(f"   {BOOT_SEED} for every comparison (bootstrap then sign flips from the same stream).")
    add("   Holm step-down adjustment within each family; significant = Holm-adjusted p < 0.05; intervals are")
    add("   not adjusted.")

    add("\n6. HOLM FAMILIES")
    add("   RQ1 (NEXA - GraphCypherQAChain)        35 = 30 AlzKB (5 configurations x 6 subsets) + 5 BioHopR")
    add("   RQ1 secondary (NEXA - LLM only)        35 = 30 + 5")
    add("   RQ2 (Flex - Direct)                    14 = 12 + 2")
    add("   RQ3 (Qwen3.8-27B - GPT-5-mini)         14 = 12 + 2")
    add("   Negation sensitivity (AlzKB only)       9 = the RQ1-RQ3 contrasts on the disambiguated TF 2-hop set,")
    add("                                            one separate family; the original TF 2-hop comparisons stay in")
    add("                                            their RQ families, unchanged (not replaced)")
    add("   NEXA vs published ESCARGOT              4 (AlzKB binary subsets, worst-case McNemar bound)")
    add("   Component ablation                      9 per benchmark at figure level (BioHopR; AlzKB pooled);")
    add("                                          54 for AlzKB per subset (9 arms x 6 subsets)")
    add("   Entity-name perturbations              12 per benchmark and metric (3 arms x 4 perturbation kinds)")

    add("\n7. SOFTWARE")
    add(f"   Python {platform.python_version()}; " + "; ".join(
        f"{p} {md.version(p)}" for p in ("numpy", "scipy", "langchain", "langchain-core", "langgraph",
                                          "langchain-openai", "langchain-ollama", "openai", "neo4j",
                                          "langchain-neo4j")))
    add("   Graphs in Neo4j 5.26.0 community with APOC: AlzKB v1.2.1, PrimeKG v2.1 (BioHopR).")
    add("   Models: gpt-4o-mini and gpt-5-mini through the OpenAI API (temperature 0 for gpt-4o-mini; gpt-5-mini")
    add("   accepts only the default temperature 1); Qwen3.8-27B (qwen3.8:27b-q8_0) served locally through Ollama,")
    add("   temperature 0.")
    add("   OpenAI model snapshots are not recorded by the API calls.")

    add("\n8. RUN CONFIGURATION AND PROVENANCE (dates: answer timestamps; commits: each run file's .meta.json)")
    add(f"   Ablations and perturbations: NEXA Direct, {MODEL}, examples_mode llm (generic-examples: hardcoded),")
    add("   pinned settings " + ", ".join(f"{k}={v}" for k, v in PINNED_ENV.items()) + ".")
    for exp in ("alzkb_ablation", "biohopr_ablation", "alzkb_perturbation", "biohopr_perturbation"):
        for d in sorted(glob.glob(os.path.join(RESULTS, exp, "*_*"))):
            add(f"   {exp}/{os.path.basename(d):38s} " + fmt_provenance(provenance(os.path.join(d, "*.jsonl"))))
    add("   Research questions RQ1-RQ3: complete benchmarks (AlzKB: 2,975 questions over the six subsets, plus the")
    add("   540 disambiguated TF 2-hop questions of the sensitivity analysis; BioHopR: 7,633), configurations")
    add("   evaluated separately from the ablations. Dates below are those of the answers (resumed and retried")
    add("   runs span several sessions).")
    for b in ("alzkb", "biohopr"):
        for d in sorted(glob.glob(os.path.join(RESULTS, b, "*_*"))):
            add(f"   {b}/{os.path.basename(d):28s} " + fmt_provenance(provenance(os.path.join(d, "*.jsonl"))))

    add("\n9. DATA")
    add("   Every answer is kept question by question (benchmark/results/<experiment>/<arm>/*.jsonl): question,")
    add("   gold, model answer, generated and final Cypher, correction counts, tokens, latency. All paired tests")
    add("   can be recomputed from these files with the scripts named above.")
    return "\n".join(L) + "\n"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    outputs = {
        "component_ablation.csv": component_ablation(),
        "entity_perturbations.csv": entity_perturbations(),
        "paired_comparisons.csv": paired_comparisons(),
    }
    for name, rows in outputs.items():
        write_csv(os.path.join(OUT, name), rows)
        print(f"{name}: {len(rows)} rows")
    with open(os.path.join(OUT, "experimental_details.txt"), "w", encoding="utf-8") as fh:
        fh.write(details())
    print(f"written to {os.path.relpath(OUT, _REPO)}/")


if __name__ == "__main__":
    main()
