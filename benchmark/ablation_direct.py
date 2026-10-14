"""Ablation of NEXA Direct: what each component of the pipeline contributes.

Pipeline (nexa/knowledge_graph/graph_query.py, query_correction.py): a one-shot
Cypher query is generated and executed. Only if it returns nothing does staged
regeneration run (rephrasing, label selection, relationship selection, Cypher
generation), followed by the correction loop: a Neo4j error goes to syntax
repair (<= 3 rounds); an empty result has its literals matched against database
values, the LLM picks a candidate or proposes a synonym and rewrites the query
(<= 3 rounds); still empty is accepted as an absence. Components act only on
these "triggered" questions, so each arm is reported on all questions and on
the triggered ones.

Arms (each changes one thing in the published configuration)
    full              the published configuration
    one-shot          no staged regeneration: the loop repairs the one-shot query
    decomposed        always the staged path, no one-shot first
    no-empty-loop     an empty result is final
    no-db-matching    the loop rewrites without database candidate values
    no-llm-selection  the top-ranked candidate is used; no LLM choice or synonym
    no-syntax         Neo4j errors are handled like an empty result
    bare              one-shot, executed once: every component removed
    no-examples       every prompt zero-shot
    generic-examples  the generic examples of prompts.py: same query shapes,
                      <LABEL>/<REL_TYPE> placeholders, nothing from the graph
Switches are NEXA_ABLATE_* (nexa/config/params.py), recorded in each run's
.meta.json; `status` checks that each arm ran with its own switches only.

The published examples are built from the graph, not the benchmark, but on
AlzKB their literals include Alzheimer's Disease and FAD, so the example arms
are also reported on the questions that cite such a name and on the others.

Experiments (GPT-4o-mini; seed 20260926; proportional stratified allocation)
    biohopr               1 000 BioHopR 2-hop questions (PrimeKG 2.1), strata
                          relation path x answer-set size. All arms.
    alzkb                 1 000 AlzKB questions over the six sets, strata set x
                          answer class. All arms, plus a pooled primary metric.
    biohopr-perturbation  150 base questions whose start (AlzKB: anchor) entity
    alzkb-perturbation    is a graph node name, perturbed: case change, one
                          letter deleted, gene symbol -> full gene name; gold
                          unchanged. Arms full, no-empty-loop, no-db-matching,
                          no-llm-selection.
The full gene name is AlzKB Gene.commonName; PrimeKG stores only the symbol,
so for BioHopR it is taken from AlzKB through the NCBI Gene id.
Each sample's manifest records its seed, its strata and a representativeness
check: each complete published arm's full-set score should fall inside the
same arm's 95% interval on the sample.

Usage (from the repository root)
    poetry run python -m benchmark.ablation_direct build      # samples and perturbation sets (done once)
    poetry run python -m benchmark.ablation_direct run --experiment alzkb-perturbation
    poetry run python -m benchmark.ablation_direct status     # progress, switch check
    poetry run python -m benchmark.ablation_direct report     # -> results/<exp>/stats/ablation.*

`run` resumes each run's own file; `--experiment` and `--arms` restrict it.
`commands` prints the same runs as shell commands. Building a perturbation set
needs its graph's Neo4j (BioHopR's also needs AlzKB's); the samples do not.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from benchmark.bench_utils import question_id, wilson_ci
from benchmark.paired_stats import N_BOOT, compare_binary, holm_adjust, mean_ci_bca, paired_bootstrap_diff
from benchmark.report_paired_stats import PRESETS, graded_score, load_arm

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = os.path.join(_REPO, "benchmark", "datasets")
RESULTS = os.path.join(_REPO, "benchmark", "results")

MODEL, PROVIDER = "gpt-4o-mini", "openai"
SEED = 20260926

# List prices (USD per 1M tokens) used only for the cost estimate printed by
# `commands`; no prompt-cache discount, so the real bill is at most this.
PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-5-mini": (0.25, 2.00)}

ABLATION_FLAGS = (
    "NEXA_ABLATE_EMPTY_RESULT_LOOP",
    "NEXA_ABLATE_DB_MATCHING",
    "NEXA_ABLATE_LLM_SELECTION",
    "NEXA_ABLATE_SYNTAX_CORRECTION",
    "NEXA_ABLATE_FEWSHOT_EXAMPLES",
)
# Pinned for every run whatever the local .env says, with the values the
# reference runs recorded in their .meta.json.
PINNED_ENV = {
    "NEXA_COMPLETENESS_GUARD": "false",
    "NEXA_NO_LIMIT_PROMPT": "false",
    "NEXA_FLEX_ANSWER_RESCUE": "false",
    "NEXA_PROPERTY_MATCHING": "composite",
    "NEXA_LLM_MAX_OUTPUT_TOKENS": "32768",
    "NEXA_LLM_MAX_PROPERTY_TOKENS": "1024",
}


@dataclass(frozen=True)
class Arm:
    """One ablation arm: generation mode, switches turned on, and what it removes."""
    name: str
    generation_mode: str
    flags: Tuple[str, ...]
    removes: str
    examples_mode: str = "llm"      # the published runs use the schema-grounded, LLM-phrased examples


ARMS: Dict[str, Arm] = {a.name: a for a in [
    Arm("full", "one_shot_fallback", (), "nothing (published configuration)"),
    Arm("one-shot", "one_shot", (), "staged regeneration: the loop repairs the one-shot query"),
    Arm("decomposed", "decomposed", (), "one-shot first: every question takes the staged path"),
    Arm("no-empty-loop", "one_shot_fallback", ("NEXA_ABLATE_EMPTY_RESULT_LOOP",), "the empty-result correction loop"),
    Arm("no-db-matching", "one_shot_fallback", ("NEXA_ABLATE_DB_MATCHING",), "database value matching"),
    Arm("no-llm-selection", "one_shot_fallback", ("NEXA_ABLATE_LLM_SELECTION",), "LLM choice among candidates and synonyms"),
    Arm("no-syntax", "one_shot_fallback", ("NEXA_ABLATE_SYNTAX_CORRECTION",), "syntax repair of Neo4j errors"),
    Arm("bare", "one_shot", ("NEXA_ABLATE_EMPTY_RESULT_LOOP", "NEXA_ABLATE_SYNTAX_CORRECTION"),
        "everything: one-shot, executed once"),
    Arm("no-examples", "one_shot_fallback", ("NEXA_ABLATE_FEWSHOT_EXAMPLES",),
        "the few-shot examples: every prompt is zero-shot"),
    Arm("generic-examples", "one_shot_fallback", (),
        "the graph's content in the examples: same shapes with <LABEL>/<REL_TYPE> placeholders",
        examples_mode="hardcoded"),
]}


def arm_dir(arm: str, model: str = MODEL, system: str = "nexa-direct") -> str:
    """Arm directory name: `<system>_<model>`, plus `_<arm>` for an ablated arm."""
    return f"{system}_{model}" + ("" if arm == "full" else f"_{arm}")


ALZKB_SETS = [
    ("True_or_False_1hop.json", "tf"), ("True_or_False_2hop.json", "tf"),
    ("MCQ_1hop.json", "mcq"), ("MCQ_2hop.json", "mcq"),
    ("OpenEnded_1hop.json", "open_ended"), ("OpenEnded_2hop.json", "open_ended"),
]


@dataclass
class Experiment:
    """One ablation experiment: question files, KG profile and arms."""
    key: str
    name: str                       # benchmark/datasets/<name>/ and benchmark/results/<name>/
    profile: str
    files: List[Tuple[str, str]]    # (file in datasets/<name>/, --type)
    arms: List[str]
    description: str = ""
    reference: Optional[str] = None  # an existing complete run on the same code, for run-to-run noise
    extra: Dict[str, Any] = field(default_factory=dict)


PERTURBATION_ARMS = ["full", "no-empty-loop", "no-db-matching", "no-llm-selection"]

EXPERIMENTS: Dict[str, Experiment] = {
    "biohopr": Experiment(
        "biohopr", "biohopr_ablation", "PrimeKG 21",
        [("BioHopR_2hop_abl1000.json", "open_ended_list")], list(ARMS),
        "BioHopR 2-hop, 1 000 stratified questions",
        reference="benchmark/results/biohopr/nexa-direct_gpt-4o-mini/*.jsonl"),
    "biohopr-perturbation": Experiment(
        "biohopr-perturbation", "biohopr_perturbation", "PrimeKG 21",
        [("BioHopR_2hop_perturbed.json", "open_ended_list")],
        PERTURBATION_ARMS,
        "BioHopR 2-hop questions with a perturbed start entity name"),
    "alzkb": Experiment(
        "alzkb", "alzkb_ablation", "Alzkb 121",
        [(f"{s[:-5]}_abl.json", t) for s, t in ALZKB_SETS], list(ARMS),
        "AlzKB, the six ESCARGOT sets, 1 000 stratified questions"),
    "alzkb-perturbation": Experiment(
        "alzkb-perturbation", "alzkb_perturbation", "Alzkb 121",
        [("OpenEnded_1hop_perturbed.json", "open_ended"), ("MCQ_1hop_perturbed.json", "mcq")],
        PERTURBATION_ARMS,
        "AlzKB 1-hop questions with a perturbed anchor entity name"),
}

# A sample file names its source set plus a suffix: MCQ_1hop_abl.json,
# BioHopR_2hop_abl1000.json, OpenEnded_1hop_perturbed.json.
_SUFFIX = re.compile(r"_(abl\d*|perturbed)(?=\.json$)")


def source_set(file_name: str) -> str:
    """Source question set of a sample file (MCQ_1hop_abl.json -> MCQ_1hop.json)."""
    return _SUFFIX.sub("", file_name or "")


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def allocate(counts: Dict[Any, int], n: int, caps: Optional[Dict[Any, int]] = None) -> Dict[Any, int]:
    """Proportional allocation of `n` draws over strata (largest remainder).

    Each stratum gets floor(n * share); leftover draws go to the most
    under-served strata, ties broken by stratum key. `caps` bounds each stratum
    (a nested sub-sample cannot exceed its parent sample).
    """
    total = sum(counts.values())
    if not 0 < n <= total:
        raise ValueError(f"cannot draw {n} from {total}")
    caps = caps if caps is not None else counts
    quota = {k: n * c / total for k, c in counts.items()}
    alloc = {k: min(int(math.floor(q)), caps[k]) for k, q in quota.items()}
    by_fraction = sorted(counts, key=lambda k: (-(quota[k] - math.floor(quota[k])), str(k)))
    while sum(alloc.values()) < n:
        room = [k for k in by_fraction if alloc[k] < caps[k]]
        if not room:
            raise ValueError("caps too tight for the requested size")
        # Most under-served stratum first, relative to its quota.
        room.sort(key=lambda k: (-(quota[k] - alloc[k]), str(k)))
        alloc[room[0]] += 1
    return alloc


def stratum_permutations(keys: Sequence[Any], seed: int) -> Dict[Any, List[int]]:
    """Index lists per stratum, each shuffled by a generator seeded from (seed, stratum).

    Other strata are unaffected when one changes, and a nested sample takes a
    prefix of the same permutation.
    """
    groups: Dict[Any, List[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)
    out = {}
    for k, idx in groups.items():
        rng = random.Random(f"{seed}|{k}")
        idx = list(idx)
        rng.shuffle(idx)
        out[k] = idx
    return out


def draw(perms: Dict[Any, List[int]], alloc: Dict[Any, int]) -> List[int]:
    """The first alloc[k] indices of each stratum, in original order."""
    return sorted(i for k, idx in perms.items() for i in idx[:alloc.get(k, 0)])


# ---------------------------------------------------------------------------
# Strata
# ---------------------------------------------------------------------------

def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def biohopr_population() -> Tuple[List[dict], List[Tuple[str, str]]]:
    """BioHopR 2-hop items and their strata (relation path, answer-set size)."""
    data = load_json(os.path.join(DATASETS, "biohopr", "BioHopR_2hop.json"))
    return data, [(d["meta"]["path"], d["meta"]["size_bucket"]) for d in data]


def _oe_size(answer: str) -> str:
    n = len([x for x in str(answer).split(",") if x.strip()])
    return "1" if n <= 1 else "2-5" if n <= 5 else "6-20" if n <= 20 else "21+"


def alzkb_stratum(set_name: str, item: dict) -> Tuple[str, str]:
    """(set, answer class): TF gold x negation, MCQ correct option, open-ended gold size."""
    q, a = item["question"], str(item["answer"]).strip()
    if set_name.startswith("True_or_False"):
        return set_name, f"{a}|{'negated' if ' not ' in q else 'affirmative'}"
    if set_name.startswith("MCQ"):
        return set_name, f"option {a}"
    return set_name, f"|gold| {_oe_size(a)}"


def alzkb_population() -> Tuple[List[Tuple[str, dict]], List[Tuple[str, str]]]:
    """Every AlzKB item as (set, item), with its stratum."""
    items = [(s, it) for s, _ in ALZKB_SETS for it in load_json(os.path.join(DATASETS, "alzkb", s))]
    return items, [alzkb_stratum(s, it) for s, it in items]


# ---------------------------------------------------------------------------
# Representativeness: the sample must reproduce the population's scores
# ---------------------------------------------------------------------------

def _primary(set_name: str) -> str:
    return "accuracy" if set_name.startswith(("True_or_False", "MCQ")) else "jaccard"


def representativeness(preset: str, sample: Dict[str, set], population: Dict[str, set]) -> Dict[str, Any]:
    """Each complete published arm scored on the whole set and on the sample.

    The population value should lie inside the sample's 95% interval for about
    95% of (arm, set) pairs.
    """
    rows = []
    for label, globs in PRESETS[preset]:
        recs = load_arm(globs, label=label)
        by_set: Dict[str, Dict[str, dict]] = defaultdict(dict)
        for (ds, qid), r in recs.items():
            by_set[ds][qid] = r
        for s, pop_qids in population.items():
            have = by_set.get(s, {})
            if not pop_qids <= set(have):
                continue            # incomplete arm on this set (e.g. a run still in progress)
            metric = _primary(s)
            val = (lambda r: float(bool(r.get("correct")))) if metric == "accuracy" else (lambda r: graded_score(r, "jaccard"))
            pop = [val(have[q]) for q in pop_qids]
            smp = [val(have[q]) for q in sample[s]]
            if metric == "accuracy":
                lo, hi = wilson_ci(int(sum(smp)), len(smp))
            else:
                lo, hi = mean_ci_bca(smp)
            p_mean, s_mean = sum(pop) / len(pop), sum(smp) / len(smp)
            rows.append({"arm": label, "set": s, "metric": metric, "population": p_mean,
                         "sample": s_mean, "ci95": [lo, hi], "inside": lo <= p_mean <= hi})
    inside = sum(r["inside"] for r in rows)
    return {"rows": rows, "inside": inside, "checked": len(rows)}


def print_representativeness(rep: Dict[str, Any]) -> None:
    print(f"  representativeness: population value inside the sample's 95% CI for "
          f"{rep['inside']}/{rep['checked']} (arm, set) pairs")
    for r in rep["rows"]:
        if not r["inside"]:
            print(f"    outside: {r['arm']:26s} {r['set']:28s} population {100 * r['population']:5.1f}  "
                  f"sample {100 * r['sample']:5.1f} [{100 * r['ci95'][0]:.1f}, {100 * r['ci95'][1]:.1f}]")


def strata_table(keys: Sequence[Any], chosen: Iterable[int]) -> List[Dict[str, Any]]:
    """Population and sample counts per stratum."""
    pop, smp = Counter(keys), Counter(keys[i] for i in chosen)
    return [{"stratum": list(k) if isinstance(k, tuple) else k, "population": pop[k], "sample": smp.get(k, 0)}
            for k in sorted(pop, key=str)]


# ---------------------------------------------------------------------------
# build: samples
# ---------------------------------------------------------------------------

def build_biohopr(n: int = 1000) -> None:
    """Draw the BioHopR sample and write it with its manifest."""
    data, keys = biohopr_population()
    perms = stratum_permutations(keys, SEED)
    chosen = draw(perms, allocate(Counter(keys), n))
    items = [data[i] for i in chosen]
    exp = EXPERIMENTS["biohopr"]
    out_dir = os.path.join(DATASETS, exp.name)
    write_json(os.path.join(out_dir, exp.files[0][0]), items)
    qids = {question_id(it["question"]) for it in items}
    rep = representativeness("biohopr", {"BioHopR_2hop.json": qids},
                             {"BioHopR_2hop.json": {question_id(d["question"]) for d in data}})
    write_json(os.path.join(out_dir, "manifest.json"), {
        "description": exp.description, "source": "benchmark/datasets/biohopr/BioHopR_2hop.json",
        "seed": SEED, "n": len(items), "population_n": len(data),
        "strata": "meta.path x meta.size_bucket, proportional allocation (largest remainder)",
        "strata_table": strata_table(keys, chosen), "qids": sorted(qids), "representativeness": rep,
    })
    print(f"biohopr: {len(items)} / {len(data)} questions -> {os.path.relpath(out_dir, _REPO)}")
    print_representativeness(rep)


def build_alzkb(n: int = 1000) -> None:
    """Draw the AlzKB sample over the six sets and write it with its manifest."""
    items, keys = alzkb_population()
    perms = stratum_permutations(keys, SEED)
    chosen = draw(perms, allocate(Counter(keys), n))
    exp = EXPERIMENTS["alzkb"]
    out_dir = os.path.join(DATASETS, exp.name)
    per_set: Dict[str, List[dict]] = defaultdict(list)
    for i in chosen:
        per_set[items[i][0]].append(items[i][1])
    for s, fname in zip([s for s, _ in ALZKB_SETS], [f for f, _ in exp.files]):
        write_json(os.path.join(out_dir, fname), per_set[s])
    sample = {s: {question_id(it["question"]) for it in its} for s, its in per_set.items()}
    population = defaultdict(set)
    for s, it in items:
        population[s].add(question_id(it["question"]))
    rep = representativeness("alzkb", sample, dict(population))
    write_json(os.path.join(out_dir, "manifest.json"), {
        "description": exp.description, "source": "benchmark/datasets/alzkb/",
        "seed": SEED, "n": len(chosen), "population_n": len(items),
        "strata": "set x answer class (True/False: gold x negated statement; MCQ: correct option; "
                  "open-ended: answer-set size), proportional allocation over all sets (largest remainder)",
        "per_set": {s: {"population": len(population[s]), "sample": len(sample.get(s, ()))} for s, _ in ALZKB_SETS},
        "strata_table": strata_table(keys, chosen),
        "qids": {s: sorted(q) for s, q in sample.items()}, "representativeness": rep,
    })
    print(f"alzkb: {len(chosen)} / {len(items)} questions -> {os.path.relpath(out_dir, _REPO)}")
    for s, _ in ALZKB_SETS:
        print(f"    {s:28s} {len(sample.get(s, ())):4d} / {len(population[s])}")
    print_representativeness(rep)


# ---------------------------------------------------------------------------
# build: perturbation set
# ---------------------------------------------------------------------------

# Anchor phrase in the question stem -> (node label, identifying property).
ANCHORS = {
    "drug class": ("DrugClass", "commonName"), "drug": ("Drug", "commonName"),
    "gene": ("Gene", "geneSymbol"), "disease": ("Disease", "commonName"),
    "symptom": ("Symptom", "commonName"), "pathway": ("Pathway", "commonName"),
    "biological process": ("BiologicalProcess", "commonName"),
    "molecular function": ("MolecularFunction", "commonName"),
    "cellular component": ("CellularComponent", "commonName"),
}
_ANCHOR_RE = re.compile(r"\bthe (" + "|".join(sorted(ANCHORS, key=len, reverse=True)) + r")\s+(.+)$")
PERTURBATIONS = ("original", "case", "typo", "alias")
N_PERTURBATION_BASE = 150


def find_anchor(question: str) -> Optional[Tuple[str, str, int, int]]:
    """(anchor type, name, start, end) of the entity the question is about."""
    stem = question.split("?")[0]
    m = _ANCHOR_RE.search(stem)
    if not m:
        return None
    name, start = m.group(2).rstrip(" ."), m.start(2)
    # "... increases the gene expression of FOXM1": the gene is after "of".
    if m.group(1) == "gene" and name.startswith("expression of "):
        name, start = name[len("expression of "):], start + len("expression of ")
    return m.group(1), name, start, start + len(name)


def perturb_case(name: str) -> Optional[str]:
    """Lower-cased name (title-cased if already lower case); None if unchanged."""
    new = name.lower() if name != name.lower() else name.title()
    return new if new != name else None


def perturb_typo(name: str, key: str) -> Optional[str]:
    """Delete one internal letter, chosen by a generator seeded from `key`; None under 5 characters."""
    pos = [i for i in range(1, len(name) - 1) if name[i].isalpha()]
    if len(name) < 5 or not pos:
        return None
    i = random.Random(f"{SEED}|typo|{key}").choice(pos)
    return name[:i] + name[i + 1:]


def _neo4j_session(profile: str):
    from neo4j import GraphDatabase
    p = load_json(os.path.join(_REPO, "api_data", "kg_profiles.json"))[profile]
    driver = GraphDatabase.driver(f"bolt://{p['host']}:{p['port']}", auth=(p["username"], p["password"]))
    return driver


def build_alzkb_perturbation(n_base: int = N_PERTURBATION_BASE) -> None:
    """Build the AlzKB perturbation set from 1-hop open-ended and MCQ questions (needs the AlzKB Neo4j)."""
    exp = EXPERIMENTS["alzkb-perturbation"]
    driver = _neo4j_session(exp.profile)
    sources = [("OpenEnded_1hop.json", exp.files[0][0]), ("MCQ_1hop.json", exp.files[1][0])]

    def exists(label: str, prop: str, value: str) -> bool:
        with driver.session() as s:
            return s.run(f"MATCH (n:`{label}`) WHERE n.`{prop}` = $v RETURN count(n) > 0 AS ok", v=value).single()["ok"]

    eligible, skipped = [], Counter()
    for src, _ in sources:
        for it in load_json(os.path.join(DATASETS, "alzkb", src)):
            a = find_anchor(it["question"])
            if not a:
                skipped["no anchor phrase"] += 1
                continue
            kind, name, start, end = a
            label, prop = ANCHORS[kind]
            if not exists(label, prop, name):
                skipped[f"'{kind}' name not in the graph"] += 1
                continue
            eligible.append((src, it, kind, name, start, end, label, prop))
    keys = [(e[0], e[2]) for e in eligible]
    chosen = draw(stratum_permutations(keys, SEED), allocate(Counter(keys), n_base))

    out: Dict[str, List[dict]] = defaultdict(list)
    manifest_items: Dict[str, dict] = {}
    dropped = Counter()
    for i in chosen:
        src, it, kind, name, start, end, label, prop = eligible[i]
        q = it["question"]
        base_qid = question_id(q)
        variants = {"original": name, "case": perturb_case(name), "typo": perturb_typo(name, base_qid)}
        if label == "Gene":
            with driver.session() as s:
                rec = s.run("MATCH (g:Gene {geneSymbol: $v}) RETURN g.commonName AS n LIMIT 1", v=name).single()
            full = rec["n"] if rec else None
            variants["alias"] = full if full and full != name else None
        for kind_p in PERTURBATIONS:
            new = variants.get(kind_p)
            if new is None:
                continue
            # A perturbed name that is another node's exact name would change the gold.
            if kind_p in ("case", "typo") and exists(label, prop, new):
                dropped[f"{kind_p}: collides with an existing name"] += 1
                continue
            new_q = q[:start] + new + q[end:]
            qid = question_id(new_q)
            if qid in manifest_items:
                continue
            out[src].append({"question": new_q, "answer": it["answer"]})
            manifest_items[qid] = {"base_qid": base_qid, "set": src, "perturbation": kind_p,
                                   "anchor_type": kind, "label": label, "property": prop,
                                   "anchor": name, "anchor_perturbed": new}
    driver.close()
    out_dir = os.path.join(DATASETS, exp.name)
    for src, fname in sources:
        write_json(os.path.join(out_dir, fname), out[src])
    counts = Counter(v["perturbation"] for v in manifest_items.values())
    write_json(os.path.join(out_dir, "manifest.json"), {
        "description": exp.description, "seed": SEED, "n_base": len(chosen), "n_eligible": len(eligible),
        "eligibility": "1-hop open-ended and MCQ questions whose anchor ('the <type> <name>') is the exact "
                       "identifying value of a node in AlzKB",
        "not_eligible": dict(skipped), "dropped": dict(dropped),
        "perturbations": {"case": "lower-cased (title-cased if already lower case)",
                          "typo": "one internal letter deleted (names of 5+ characters)",
                          "alias": "gene symbol replaced by the gene's full name (Gene.commonName)"},
        "strata": "source set x anchor type, proportional allocation over eligible questions",
        "counts": dict(counts), "items": manifest_items,
    })
    print(f"perturbation: {len(chosen)} base questions (of {len(eligible)} eligible) -> "
          f"{sum(len(v) for v in out.values())} items {dict(counts)} -> {os.path.relpath(out_dir, _REPO)}")
    if dropped:
        print(f"    dropped: {dict(dropped)}")


# BioHopR start types -> PrimeKG label (every PrimeKG node is identified by `name`).
BIOHOPR_LABELS = {"drug": "Drug", "disease": "Disease", "gene/protein": "GeneProtein",
                  "effect/phenotype": "EffectPhenotype"}


def locate_start(question: str, name: str) -> Optional[Tuple[int, int]]:
    """Span of the start entity at the end of a BioHopR question, or None if it is not there."""
    stem = question.rstrip()
    end = len(stem) - 1 if stem.endswith((".", "?")) else len(stem)
    start = end - len(name)
    if start <= 0 or stem[start:end] != name or stem[start - 1] != " ":
        return None
    return start, end


def build_biohopr_perturbation(n_base: int = N_PERTURBATION_BASE) -> None:
    """Build the BioHopR perturbation set with the AlzKB procedure (needs both Neo4j instances).

    Same eligibility (the start entity is a node's exact name), perturbations,
    collision check, base size and seed.
    """
    exp = EXPERIMENTS["biohopr-perturbation"]
    driver = _neo4j_session(exp.profile)
    alzkb = _neo4j_session(EXPERIMENTS["alzkb"].profile)

    def exists(label: str, value: str) -> bool:
        with driver.session() as s:
            return s.run(f"MATCH (n:`{label}`) WHERE n.name = $v RETURN count(n) > 0 AS ok", v=value).single()["ok"]

    def full_gene_name(symbol: str) -> Optional[str]:
        """The NCBI full name, through AlzKB, of the one PrimeKG gene with this symbol."""
        with driver.session() as s:
            ids = s.run("MATCH (g:GeneProtein {name: $v}) RETURN collect(DISTINCT g.id) AS ids", v=symbol).single()["ids"]
        if len(ids) != 1 or not str(ids[0]).isdigit():
            return None
        with alzkb.session() as s:
            rows = s.run("MATCH (g:Gene) WHERE g.xrefNcbiGene = toInteger($id) RETURN DISTINCT g.commonName AS n",
                         id=str(ids[0])).data()
        names = [r["n"] for r in rows if r["n"]]
        return names[0] if len(names) == 1 else None

    eligible, skipped = [], Counter()
    for it in load_json(os.path.join(DATASETS, "biohopr", "BioHopR_2hop.json")):
        meta = it["meta"]
        name, kind = meta["start"], meta["start_type"]
        span = locate_start(it["question"], name)
        label = BIOHOPR_LABELS.get(kind)
        if span is None:
            skipped["start entity not at the end of the question"] += 1
            continue
        if label is None or not exists(label, name):
            skipped[f"'{kind}' name not in the graph"] += 1
            continue
        eligible.append((it, kind, name, span, label))
    keys = [e[0]["meta"]["path"] for e in eligible]
    chosen = draw(stratum_permutations(keys, SEED), allocate(Counter(keys), n_base))

    items: List[dict] = []
    manifest_items: Dict[str, dict] = {}
    dropped = Counter()
    for i in chosen:
        it, kind, name, (start, end), label = eligible[i]
        q = it["question"]
        base_qid = question_id(q)
        variants = {"original": name, "case": perturb_case(name), "typo": perturb_typo(name, base_qid)}
        if label == "GeneProtein":
            full = full_gene_name(name)
            variants["alias"] = full if full and full != name else None
        for kind_p in PERTURBATIONS:
            new = variants.get(kind_p)
            if new is None:
                continue
            if kind_p in ("case", "typo") and exists(label, new):
                dropped[f"{kind_p}: collides with an existing name"] += 1
                continue
            new_q = q[:start] + new + q[end:]
            qid = question_id(new_q)
            if qid in manifest_items:
                continue
            items.append({"question": new_q, "answer": it["answer"],
                          "meta": {**it["meta"], "perturbation": kind_p, "start_perturbed": new}})
            manifest_items[qid] = {"base_qid": base_qid, "set": "BioHopR_2hop.json", "perturbation": kind_p,
                                   "anchor_type": kind, "label": label, "property": "name",
                                   "anchor": name, "anchor_perturbed": new}
    driver.close()
    alzkb.close()
    out_dir = os.path.join(DATASETS, exp.name)
    write_json(os.path.join(out_dir, exp.files[0][0]), items)
    counts = Counter(v["perturbation"] for v in manifest_items.values())
    write_json(os.path.join(out_dir, "manifest.json"), {
        "description": exp.description, "seed": SEED, "n_base": len(chosen), "n_eligible": len(eligible),
        "eligibility": "BioHopR 2-hop questions whose start entity, which ends the question, is the exact "
                       "name of a node of its type in PrimeKG",
        "not_eligible": dict(skipped), "dropped": dict(dropped),
        "perturbations": {"case": "lower-cased (title-cased if already lower case)",
                          "typo": "one internal letter deleted (names of 5+ characters)",
                          "alias": "gene symbol replaced by the gene's NCBI full name, taken from AlzKB "
                                   "(Gene.xrefNcbiGene = PrimeKG GeneProtein.id); PrimeKG itself stores no "
                                   "full name"},
        "strata": "relation path (which fixes the start type), proportional allocation over eligible questions",
        "base_by_start_type": dict(Counter(eligible[i][1] for i in chosen)),
        "counts": dict(counts), "items": manifest_items,
    })
    print(f"biohopr-perturbation: {len(chosen)} base questions (of {len(eligible)} eligible) -> "
          f"{len(items)} items {dict(counts)} -> {os.path.relpath(out_dir, _REPO)}")
    if dropped:
        print(f"    dropped: {dict(dropped)}")


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def _mean_usage(pattern: str) -> Tuple[float, float]:
    ins, outs = [], []
    for path in glob.glob(os.path.join(_REPO, pattern)):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                u = (json.loads(line).get("usage") or {}) if line.strip() else {}
                if u:
                    ins.append(u.get("input_tokens", 0))
                    outs.append(u.get("output_tokens", 0))
    return (sum(ins) / len(ins), sum(outs) / len(outs)) if ins else (0.0, 0.0)


def n_items(exp: Experiment) -> int:
    """Questions in an experiment's built sample files."""
    return sum(len(load_json(os.path.join(DATASETS, exp.name, f))) for f, _ in exp.files
               if os.path.exists(os.path.join(DATASETS, exp.name, f)))


def env_dict(flags: Sequence[str], model: str, provider: str) -> Dict[str, str]:
    """Run environment: model, pinned settings, and every NEXA_ABLATE_* switch on or off."""
    env = {"BENCHMARK_PROVIDER": provider, "BENCHMARK_MODEL_NAME": model, **PINNED_ENV}
    env.update({f: ("true" if f in flags else "false") for f in ABLATION_FLAGS})
    return env


def direct_argv(exp: Experiment, arm: Arm, fname: str, btype: str) -> List[str]:
    """Runner arguments of one (experiment, arm, question file) run."""
    return ["--type", btype, "--input", f"benchmark/datasets/{exp.name}/{fname}",
            "--agent-mode", "direct", "--profile", exp.profile, "--examples-mode", arm.examples_mode,
            "--generation-mode", arm.generation_mode, "--run-tag", "run1",
            "--out-subdir", f"{exp.name}/{arm_dir(arm.name)}"]


def shell_command(env: Dict[str, str], argv: Sequence[str]) -> str:
    """The same run as a copy-pasteable shell command."""
    import shlex
    return (f"env {' '.join(f'{k}={v}' for k, v in env.items())} \\\n"
            f"  poetry run python -m benchmark.benchmark {' '.join(shlex.quote(a) for a in argv)}")


def planned_runs(experiments: Sequence[str], arms: Optional[Sequence[str]]) -> List[Tuple[str, str, Dict[str, str], List[str]]]:
    """(experiment, arm, env, runner argv) for every run, in execution order."""
    runs = []
    for key in experiments:
        exp = EXPERIMENTS[key]
        for a in exp.arms:
            if arms and a not in arms:
                continue
            for fname, btype in exp.files:
                runs.append((key, a, env_dict(ARMS[a].flags, MODEL, PROVIDER), direct_argv(exp, ARMS[a], fname, btype)))
    return runs


def execute(runs: Sequence[Tuple[str, str, Dict[str, str], List[str]]]) -> None:
    """Run the runner once per (experiment, arm, question set), in order.

    Each run resumes its own output file, so re-running continues where it
    stopped. A failing run stops the sequence.
    """
    import subprocess
    for i, (key, arm, env, argv) in enumerate(runs, 1):
        print(f"\n[{i}/{len(runs)}] {key} / {arm}: {argv[3]}", flush=True)
        code = subprocess.call([sys.executable, "-m", "benchmark.benchmark", *argv],
                               cwd=_REPO, env={**os.environ, **env})
        if code != 0:
            raise SystemExit(f"run {i} failed (exit {code}); fix the cause and re-run the same command to resume")
    print("\nall runs done; next: `status`, then `report`")


# Published runs whose mean token usage gives each experiment's cost estimate.
COST_PROXY = {"biohopr": "benchmark/results/biohopr/nexa-direct_gpt-4o-mini/*.jsonl",
              "biohopr-perturbation": "benchmark/results/biohopr/nexa-direct_gpt-4o-mini/*.jsonl",
              "alzkb": "benchmark/results/alzkb/nexa-direct_gpt-4o-mini/*.jsonl",
              "alzkb-perturbation": "benchmark/results/alzkb/nexa-direct_gpt-4o-mini/*.jsonl"}


def commands(experiments: Sequence[str], arms: Optional[Sequence[str]]) -> None:
    """Print the runs as a shell script, with a cost estimate."""
    print("#!/usr/bin/env bash")
    print("# NEXA Direct ablation runs, generated by `python -m benchmark.ablation_direct commands`.")
    print("# Run from the repository root. Resumable: re-running skips questions already answered.")
    print("set -euo pipefail")
    print('[ -f benchmark/benchmark.py ] || { echo "run from the repository root" >&2; exit 1; }')
    total = 0.0
    for key in experiments:
        exp = EXPERIMENTS[key]
        n = n_items(exp)
        if not n:
            print(f"\n# {key}: sample missing; run `python -m benchmark.ablation_direct build` first")
            continue
        chosen = [a for a in exp.arms if not arms or a in arms]
        i, o = _mean_usage(COST_PROXY[key])
        per_arm = n * (i * PRICES[MODEL][0] + o * PRICES[MODEL][1]) / 1e6
        total += per_arm * len(chosen)
        print(f"\n# ===== {key}: {exp.description} ({n} questions, profile {exp.profile}; "
              f"needs that Neo4j running)")
        print(f"# estimated cost ~${per_arm:.2f} per arm, ~${per_arm * len(chosen):.2f} for {len(chosen)} arms")
        for a in chosen:
            arm = ARMS[a]
            print(f"\n# --- {key} / {a}: removes {arm.removes}")
            for fname, btype in exp.files:
                print(shell_command(env_dict(arm.flags, MODEL, PROVIDER), direct_argv(exp, arm, fname, btype)))
    print(f"\n# total estimated cost ~${total:.2f} (list prices, no prompt-cache discount)")
    print("echo 'done. Next: poetry run python -m benchmark.ablation_direct status && ... report'")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def arm_glob(exp_name: str, directory: str) -> str:
    return os.path.join("benchmark", "results", exp_name, directory, "*.jsonl")


def _read_records(pattern: str) -> List[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(_REPO, pattern))):
        with open(path, encoding="utf-8") as fh:
            out += [json.loads(line) for line in fh if line.strip()]
    return out


def check_switches(exp_name: str, directory: str, expected_flags: Sequence[str],
                   expected_mode: Optional[str], model: str = MODEL,
                   expected_examples: Optional[str] = None) -> List[str]:
    """Every .meta.json of the arm must show exactly the arm's own switches."""
    problems = []
    for path in glob.glob(os.path.join(_REPO, "benchmark", "results", exp_name, directory, "*.meta.json")):
        m = load_json(path)
        flags = m.get("flags", {})
        on = {f for f in ABLATION_FLAGS if flags.get(f)}
        if on != set(expected_flags):
            problems.append(f"{os.path.basename(path)}: switches on {sorted(on)}, expected {sorted(expected_flags)}")
        mode = (m.get("config") or {}).get("generation_mode")
        if expected_mode and (m.get("config") or {}).get("agent_mode") == "direct" and mode != expected_mode:
            problems.append(f"{os.path.basename(path)}: generation_mode {mode}, expected {expected_mode}")
        examples = (m.get("config") or {}).get("examples_mode")
        if expected_examples and examples != expected_examples:
            problems.append(f"{os.path.basename(path)}: examples_mode {examples}, expected {expected_examples}")
        if m.get("model") and m["model"] != model:
            problems.append(f"{os.path.basename(path)}: model {m['model']}, expected {model}")
    return problems


def status(experiments: Sequence[str]) -> None:
    """Print each arm's progress, errors and switch check."""
    for key in experiments:
        exp = EXPERIMENTS[key]
        expected = n_items(exp)
        print(f"\n{key} ({exp.name}, {expected} questions)")
        for a in exp.arms:
            d = arm_dir(a)
            recs = _read_records(arm_glob(exp.name, d))
            done = len({question_id(r["question"]) for r in recs if r.get("question")})
            errs = sum(1 for r in recs if r.get("error"))
            probs = check_switches(exp.name, d, ARMS[a].flags, ARMS[a].generation_mode,
                                   expected_examples=ARMS[a].examples_mode)
            state = "done" if done >= expected else ("not started" if not done else "running/partial")
            print(f"  {a:18s} {done:5d}/{expected:<5d} errors {errs:3d}  {state}"
                  + ("" if not probs else "  !! " + "; ".join(probs)))


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def load_experiment_arm(exp_name: str, directory: str, model: str = MODEL,
                        system: str = "nexa-direct") -> Dict[Tuple[str, str], dict]:
    """Records keyed by (source set, qid)."""
    recs = load_arm([arm_glob(exp_name, directory)], label=f"{system}_{model}")
    return {(source_set(ds), qid): r for (ds, qid), r in recs.items()}


def restrict(records: Dict[Tuple[str, str], dict], keys: Iterable[Tuple[str, str]]) -> Dict[Tuple[str, str], dict]:
    keys = set(keys)
    return {k: v for k, v in records.items() if k in keys}


def is_binary(set_name: str) -> bool:
    return set_name.startswith(("True_or_False", "MCQ"))


def compare(arm: Dict, ref: Dict, keys: Sequence, metric: str) -> Dict[str, Any]:
    """arm - ref on the `keys` present in both: McNemar/Newcombe for accuracy, paired BCa otherwise."""
    keys = [k for k in keys if k in arm and k in ref]
    if metric == "accuracy":
        c = compare_binary([bool(arm[k].get("correct")) for k in keys], [bool(ref[k].get("correct")) for k in keys])
        return {"n": c["n"], "a": c["acc_a"], "b": c["acc_b"], "delta": c["delta"], "ci": [c["ci_lo"], c["ci_hi"]],
                "gained": c["b_only_a"], "lost": c["c_only_b"], "p": c["p"], "test": c["test"]}
    c = paired_bootstrap_diff([graded_score(arm[k], metric) for k in keys],
                              [graded_score(ref[k], metric) for k in keys], n_boot=N_BOOT)
    gained = sum(1 for k in keys if graded_score(arm[k], metric) > graded_score(ref[k], metric))
    lost = sum(1 for k in keys if graded_score(arm[k], metric) < graded_score(ref[k], metric))
    return {"n": c["n"], "a": c["mean_a"], "b": c["mean_b"], "delta": c["delta"], "ci": [c["ci_lo"], c["ci_hi"]],
            "gained": gained, "lost": lost, "p": c["p"], "test": c["test"]}


def example_literals(profile: str) -> set:
    """Entity names used as Cypher literals in the profile's cached few-shot examples."""
    path = os.path.join(_REPO, "api_data", "kg_examples", profile.replace(" ", "_") + ".json")
    if not os.path.exists(path):
        return set()
    cache = load_json(path)
    text = "\n".join(v for block in ("examples_llm", "examples_template")
                     for v in (cache.get(block) or {}).values() if isinstance(v, str))
    # Only values used as Cypher literals: {prop: "v"}, = "v", IN ["v", ...],
    # double- or single-quoted (a double-quoted value may hold an apostrophe).
    q = r"""(?:"([^"\n]{2,80})"|'([^'\n]{2,80})')"""
    hits = re.findall(r"\{\s*\w+\s*:\s*" + q + r"\s*\}", text) + re.findall(r"=\s*" + q, text)
    for group in re.findall(r"\bIN\s*\[([^\]\n]+)\]", text):
        hits += re.findall(q, group)
    out = {a or b for a, b in hits}
    return {v.strip() for v in out if v.strip()}


def cites_any(question: str, literals: Iterable[str]) -> bool:
    """Whole-word occurrence, so that TBP does not match TBPL1."""
    return any(re.search(r"(?<!\w)" + re.escape(lit) + r"(?!\w)", question or "") for lit in literals)


def usage_summary(records: Dict) -> Dict[str, float]:
    """Mean tokens and latency per question."""
    tok = [(r.get("usage") or {}).get("total_tokens", 0) for r in records.values()]
    lat = [r.get("latency_ms") or 0 for r in records.values()]
    n = max(len(records), 1)
    return {"tokens_per_q": sum(tok) / n, "latency_s_per_q": sum(lat) / n / 1000}


def primary_score(key: Tuple[str, str], record: dict) -> float:
    """A question's primary metric: 0/1 for True/False and MCQ, Jaccard otherwise."""
    return float(bool(record.get("correct"))) if is_binary(key[0]) else graded_score(record, "jaccard")


def compare_pooled(arm: Dict, ref: Dict, keys: Sequence) -> Dict[str, Any]:
    """arm - ref on the pooled primary metric, paired BCa bootstrap over questions."""
    keys = [k for k in keys if k in arm and k in ref]
    if not keys:
        return {}
    a = [primary_score(k, arm[k]) for k in keys]
    b = [primary_score(k, ref[k]) for k in keys]
    c = paired_bootstrap_diff(a, b, n_boot=N_BOOT)
    return {"n": c["n"], "a": c["mean_a"], "b": c["mean_b"], "delta": c["delta"], "ci": [c["ci_lo"], c["ci_hi"]],
            "gained": sum(x > y for x, y in zip(a, b)), "lost": sum(x < y for x, y in zip(a, b)),
            "p": c["p"], "test": c["test"]}


def fmt_row(label: str, c: Dict[str, Any], p_holm: Optional[float] = None) -> str:
    if not c or not c.get("n"):
        return f"    {label:34s}   (no question in this group)"
    ph = "" if p_holm is None else f"  p Holm {p_holm:8.2e}{' *' if p_holm < 0.05 else ''}"
    return (f"    {label:34s} n={c['n']:4d}  {100 * c['a']:5.1f} vs {100 * c['b']:5.1f}  "
            f"Δ {100 * c['delta']:+5.1f} [{100 * c['ci'][0]:+5.1f}, {100 * c['ci'][1]:+5.1f}]  "
            f"+{c['gained']}/-{c['lost']}  p {c['p']:8.2e}{ph}")


def report_ablation(key: str) -> Optional[Dict[str, Any]]:
    """Every arm against `full`, per set, on all and on triggered questions; writes stats/ablation.*."""
    exp = EXPERIMENTS[key]
    arms = {a: load_experiment_arm(exp.name, arm_dir(a)) for a in exp.arms}
    arms = {a: r for a, r in arms.items() if r}
    if "full" not in arms:
        print(f"\n{key}: the full arm has no records yet; nothing to compare against")
        return None
    full = arms["full"]
    literals = example_literals(exp.profile)
    cites = {k for k, r in full.items() if cites_any(r.get("question"), literals)}
    sets = sorted({s for s, _ in full}, key=lambda s: [x for x, _ in ALZKB_SETS].index(s)
                  if s in dict(ALZKB_SETS) else 99)
    out: Dict[str, Any] = {"experiment": exp.name, "arms": {}, "noise": None}
    family = []
    lines = [f"\n{'=' * 110}\n{key}: {exp.description}\n{'=' * 110}",
             "  Δ = arm − full on the same questions (pp). +gained/−lost: questions the arm gets right/wrong "
             "that full does not.",
             "  triggered = questions whose one-shot query came back empty in the full run: the only ones "
             "on which a component acts."]
    for a, recs in arms.items():
        if a == "full":
            continue
        arm_out = {"removes": ARMS[a].removes, "sets": {}, "usage": usage_summary(recs)}
        for s in sets:
            keys = [k for k in full if k[0] == s]
            metric = "accuracy" if is_binary(s) else "jaccard"
            trig = [k for k in keys if full[k].get("fallback_triggered")]
            row = {"metric": metric, "all": compare(recs, full, keys, metric),
                   "triggered": compare(recs, full, trig, metric),
                   "not_triggered": compare(recs, full, [k for k in keys if k not in set(trig)], metric)}
            if not is_binary(s):
                row["exact_match_all"] = compare(recs, full, keys, "accuracy")
            if s.startswith("True_or_False"):
                for pol in ("True", "False"):
                    pk = [k for k in keys if str(full[k].get("expected")).strip() == pol]
                    row[f"gold_{pol}"] = compare(recs, full, pk, metric)
            if a.endswith("examples") and literals:
                row["cites_example_entity"] = compare(recs, full, [k for k in keys if k in cites], metric)
                row["no_example_entity"] = compare(recs, full, [k for k in keys if k not in cites], metric)
            arm_out["sets"][s] = row
            if row["all"].get("n"):
                family.append((a, s, row["all"]["p"]))
        out["arms"][a] = arm_out
    adj = dict(zip([(a, s) for a, s, _ in family], holm_adjust([p for _, _, p in family])))
    out["holm_family"] = len(family)

    base_use = usage_summary(full)
    for a, arm_out in out["arms"].items():
        u = arm_out["usage"]
        lines.append(f"\n  --- {a}: removes {arm_out['removes']}   "
                     f"(tokens/question {u['tokens_per_q']:.0f} vs {base_use['tokens_per_q']:.0f} full; "
                     f"latency {u['latency_s_per_q']:.1f}s vs {base_use['latency_s_per_q']:.1f}s)")
        for s, row in arm_out["sets"].items():
            m = "acc" if row["metric"] == "accuracy" else "Jaccard"
            row["all"]["p_holm"] = adj.get((a, s))
            lines.append(fmt_row(f"{s} {m}", row["all"], adj.get((a, s))))
            lines.append(fmt_row("   triggered", row["triggered"]))
            if "exact_match_all" in row:
                lines.append(fmt_row("   exact set match", row["exact_match_all"]))
            for pol in ("True", "False"):
                if f"gold_{pol}" in row:
                    lines.append(fmt_row(f"   gold = {pol}", row[f"gold_{pol}"]))
            if "cites_example_entity" in row:
                lines.append(fmt_row("   cites an example entity", row["cites_example_entity"]))
                lines.append(fmt_row("   no example entity", row["no_example_entity"]))

    trig_rate = {s: sum(1 for k in full if k[0] == s and full[k].get("fallback_triggered")) /
                 max(sum(1 for k in full if k[0] == s), 1) for s in sets}
    out["full_trigger_rate"] = trig_rate
    lines.append("\n  one-shot came back empty in the full run (fallback triggered): " +
                 ", ".join(f"{s} {100 * v:.1f}%" for s, v in trig_rate.items()))

    if exp.reference:
        ref = {(source_set(ds), qid): r for (ds, qid), r in
               load_arm([exp.reference], label=f"nexa-direct_{MODEL}").items()}
        ref = restrict(ref, full.keys())
        if ref:
            noise = {s: compare(full, ref, [k for k in full if k[0] == s], "accuracy" if is_binary(s) else "jaccard")
                     for s in sets}
            out["noise"] = noise
            lines.append("\n  run-to-run reference: the full arm against the earlier complete run of the same "
                         "code on the same questions")
            for s, c in noise.items():
                lines.append(fmt_row(f"{s} (full − reference run)", c))
    if len(sets) > 1:
        # Pooled over sets: each question keeps its own primary metric.
        pooled_family = []
        for a in out["arms"]:
            keys = list(full)
            trig = [k for k in keys if full[k].get("fallback_triggered")]
            row = {"all": compare_pooled(arms[a], full, keys), "triggered": compare_pooled(arms[a], full, trig)}
            if a.endswith("examples") and literals:
                row["cites_example_entity"] = compare_pooled(arms[a], full, [k for k in keys if k in cites])
                row["no_example_entity"] = compare_pooled(arms[a], full, [k for k in keys if k not in cites])
            out["arms"][a]["pooled"] = row
            pooled_family.append((a, row["all"]["p"]))
        padj = dict(zip([a for a, _ in pooled_family], holm_adjust([p for _, p in pooled_family])))
        out["pooled_holm_family"] = len(pooled_family)
        lines.append("\n  pooled over the sets: each question's primary metric (accuracy for TF and MCQ, Jaccard "
                     f"otherwise), paired bootstrap; Holm over {len(pooled_family)} arms")
        for a in out["arms"]:
            row = out["arms"][a]["pooled"]
            row["all"]["p_holm"] = padj.get(a)
            lines.append(fmt_row(f"{a}", row["all"], padj.get(a)))
            lines.append(fmt_row("   triggered", row["triggered"]))
            if "cites_example_entity" in row:
                lines.append(fmt_row("   cites an example entity", row["cites_example_entity"]))
                lines.append(fmt_row("   no example entity", row["no_example_entity"]))
    lines.append(f"\n  * Holm-adjusted p < 0.05 within this experiment's family of {len(family)} comparisons "
                 "(all questions, primary metric).")
    text = "\n".join(lines)
    print(text)
    stats_dir = os.path.join(RESULTS, exp.name, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    with open(os.path.join(stats_dir, "ablation.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    write_json(os.path.join(stats_dir, "ablation.json"), out)
    return out


def report_perturbation(key: str) -> Optional[Dict[str, Any]]:
    """Accuracy under each perturbation per arm, and each arm against full; writes stats/ablation.*."""
    exp = EXPERIMENTS[key]
    man_path = os.path.join(DATASETS, exp.name, "manifest.json")
    if not os.path.exists(man_path):
        print(f"\n{key}: set not built")
        return None
    items = load_json(man_path)["items"]
    arms = {a: {qid: r for (_, qid), r in load_experiment_arm(exp.name, arm_dir(a)).items()} for a in exp.arms}
    arms = {a: r for a, r in arms.items() if r}
    if "full" not in arms:
        print(f"\n{key}: the full arm has no records yet")
        return None
    by_pert: Dict[str, List[str]] = defaultdict(list)
    for qid, meta in items.items():
        by_pert[meta["perturbation"]].append(qid)
    orig_of = {meta["base_qid"]: qid for qid, meta in items.items() if meta["perturbation"] == "original"}
    out: Dict[str, Any] = {"table": {}, "vs_full": {}, "vs_original": {}}
    lines = [f"\n{'=' * 110}\n{key}: {exp.description}\n{'=' * 110}",
             "  accuracy (exact answer) per perturbation" + ("; open-ended and MCQ pooled" if len(exp.files) > 1 else "")]
    lines.append("    " + f"{'arm':18s}" + "".join(f"{p:>16s}" for p in PERTURBATIONS))
    for a, recs in arms.items():
        cells = []
        for p in PERTURBATIONS:
            qs = [q for q in by_pert.get(p, []) if q in recs]
            acc = sum(1 for q in qs if recs[q].get("correct")) / len(qs) if qs else float("nan")
            out["table"].setdefault(a, {})[p] = {"n": len(qs), "accuracy": acc}
            cells.append(f"{100 * acc:9.1f} (n={len(qs):3d})" if qs else f"{'—':>16s}")
        lines.append("    " + f"{a:18s}" + "".join(f"{c:>16s}" for c in cells))

    graded = {a: {q: r for q, r in recs.items() if r.get("metrics")} for a, recs in arms.items()}
    if any(graded.values()):
        lines.append("\n  Jaccard (open-ended items) per perturbation")
        for a, recs in graded.items():
            cells = []
            for p in PERTURBATIONS:
                qs = [q for q in by_pert.get(p, []) if q in recs]
                jac = sum(graded_score(recs[q], "jaccard") for q in qs) / len(qs) if qs else float("nan")
                out.setdefault("jaccard", {}).setdefault(a, {})[p] = {"n": len(qs), "jaccard": jac}
                cells.append(f"{100 * jac:9.1f} (n={len(qs):3d})" if qs else f"{'—':>16s}")
            lines.append("    " + f"{a:18s}" + "".join(f"{c:>16s}" for c in cells))

    # Gene symbols apart: a one-letter edit often yields another real symbol.
    lines.append("\n  accuracy by anchor type: gene symbols vs every other name (descriptive)")
    for a, recs in arms.items():
        for p in PERTURBATIONS:
            qs = [q for q in by_pert.get(p, []) if q in recs]
            if not qs:
                continue
            g = [q for q in qs if items[q]["label"] in ("Gene", "GeneProtein")]
            o = [q for q in qs if q not in set(g)]
            acc = lambda xs: sum(1 for q in xs if recs[q].get("correct")) / len(xs) if xs else float("nan")
            out.setdefault("by_anchor", {}).setdefault(a, {})[p] = {"gene": {"n": len(g), "accuracy": acc(g)},
                                                                     "other": {"n": len(o), "accuracy": acc(o)}}
            lines.append(f"    {a:18s} {p:9s} genes {100 * acc(g):5.1f} (n={len(g):3d})   "
                         f"other names {100 * acc(o):5.1f} (n={len(o):3d})")

    lines.append("\n  each perturbation against the original question, same arm (paired by base question)")
    for a, recs in arms.items():
        for p in PERTURBATIONS[1:]:
            pairs = [(q, orig_of.get(items[q]["base_qid"])) for q in by_pert.get(p, [])]
            pairs = [(q, o) for q, o in pairs if o and q in recs and o in recs]
            if not pairs:
                continue
            c = compare_binary([bool(recs[q].get("correct")) for q, _ in pairs],
                               [bool(recs[o].get("correct")) for _, o in pairs])
            row = {"n": c["n"], "a": c["acc_a"], "b": c["acc_b"], "delta": c["delta"], "ci": [c["ci_lo"], c["ci_hi"]],
                   "gained": c["b_only_a"], "lost": c["c_only_b"], "p": c["p"]}
            out["vs_original"].setdefault(a, {})[p] = row
            lines.append(fmt_row(f"{a} · {p} − original", row))

    family = []
    for a, recs in arms.items():
        if a == "full":
            continue
        for p in PERTURBATIONS:
            keys = [q for q in by_pert.get(p, []) if q in recs and q in arms["full"]]
            row = compare({("x", q): recs[q] for q in keys}, {("x", q): arms["full"][q] for q in keys},
                          [("x", q) for q in keys], "accuracy")
            out["vs_full"].setdefault(a, {})[p] = row
            if row.get("n"):
                family.append((a, p, row["p"]))
    adj = dict(zip([(a, p) for a, p, _ in family], holm_adjust([p for _, _, p in family])))
    lines.append(f"\n  each arm against full, per perturbation (Holm over {len(family)} comparisons)")
    for a, rows in out["vs_full"].items():
        for p, row in rows.items():
            row["p_holm"] = adj.get((a, p))
            lines.append(fmt_row(f"{a} − full · {p}", row, adj.get((a, p))))
    text = "\n".join(lines)
    print(text)
    stats_dir = os.path.join(RESULTS, exp.name, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    with open(os.path.join(stats_dir, "ablation.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    write_json(os.path.join(stats_dir, "ablation.json"), out)
    return out


# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("build", "run", "commands", "status", "report"))
    ap.add_argument("--experiment", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    ap.add_argument("--arms", nargs="+", choices=list(ARMS), help="run/commands: only these arms")
    args = ap.parse_args(argv)
    if args.action == "build":
        if "biohopr" in args.experiment:
            build_biohopr()
        if "alzkb" in args.experiment:
            build_alzkb()
        if "biohopr-perturbation" in args.experiment:
            build_biohopr_perturbation()
        if "alzkb-perturbation" in args.experiment:
            build_alzkb_perturbation()
    elif args.action == "run":
        execute(planned_runs(args.experiment, args.arms))
    elif args.action == "commands":
        commands(args.experiment, args.arms)
    elif args.action == "status":
        status(args.experiment)
    else:
        for key in args.experiment:
            report_perturbation(key) if key.endswith("perturbation") else report_ablation(key)


if __name__ == "__main__":
    sys.exit(main())
