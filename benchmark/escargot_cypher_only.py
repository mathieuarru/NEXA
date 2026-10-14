"""Per-question records and scores of the ESCARGOT Cypher-only ablation.

The released ESCARGOT implementation was re-run outside this repository with
Cypher as its only knowledge access (AlzKB 1.2.1, GPT-4o-mini). Its raw outputs:

  True/False   free text ending in an answer ("Answer: False")
  MCQ          free text, plus the option an LLM extractor (gpt-4o-mini,
               temperature 0) read in it (`openai_choice`, *_openai.jsonl)
  open-ended   the raw Cypher result: a dict, list or (rows, query) tuple

`convert` writes one record per question, in the shape of every other arm, to
benchmark/results/alzkb_escargot_cypher_only/escargot-cypher_gpt-4o-mini/,
keeping the raw answer and how it was read. `report` scores the records with
the shared scorer and writes stats/summary.json, the source of
benchmark/published_results.ESCARGOT_CYPHER_ONLY.

Reading rules:
  True/False   the first true/false after the last "Answer:", else the last
               true/false of the text; none: wrong.
  MCQ          the extractor's option, only when the response has answer text
               (not empty, not an unfinished XML/code plan) and its answer
               statement names a single option; otherwise wrong.
  open-ended   the prediction split on commas like the gold, names compared up
               to typography; an unparseable result scores 0.

MCQ 1-hop has 497 records: in the copy of the question set this run used,
entry 266 repeated the Muscle Rigidity question, so the Paraparesis, Spastic
question was never asked.

    poetry run python -m benchmark.escargot_cypher_only convert --source <escargot directory>
    poetry run python -m benchmark.escargot_cypher_only report
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from benchmark.bench_utils import compute_set_metrics, csv_to_set, question_id
    from benchmark.report_paired_stats import graded_score, load_arm
except ImportError:  # running as a script from inside benchmark/
    from bench_utils import compute_set_metrics, csv_to_set, question_id
    from report_paired_stats import graded_score, load_arm

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT = os.path.join(_REPO, "benchmark", "results", "alzkb_escargot_cypher_only")
ARM = os.path.join(EXPERIMENT, "escargot-cypher_gpt-4o-mini")
DATASETS = os.path.join(_REPO, "benchmark", "datasets", "alzkb")
SETS = {
    "True_or_False_1hop.json": "tf", "True_or_False_2hop.json": "tf",
    "MCQ_1hop.json": "mcq", "MCQ_2hop.json": "mcq",
    "OpenEnded_1hop.json": "open_ended", "OpenEnded_2hop.json": "open_ended",
}
SET_METRICS = ["exact_match", "precision", "recall", "f1", "jaccard", "overlap"]


# ---------------------------------------------------------------------------
# Reading the raw outputs
# ---------------------------------------------------------------------------

def tf_answer(text: str) -> Tuple[Optional[bool], str]:
    """(answer, how it was read) of a True/False response; None when it gives no answer."""
    t = str(text or "")
    blocks = list(re.finditer(r"answer\s*[:：]", t, re.I))
    if blocks:
        words = re.findall(r"\b(true|false)\b", t[blocks[-1].end():], re.I)
        if words:
            return words[0].lower() == "true", "answer block"
    words = re.findall(r"\b(true|false)\b", t, re.I)
    if words:
        return words[-1].lower() == "true", "last true/false of the text"
    return None, "no answer"


def has_answer_text(text: str) -> bool:
    """Whether a response holds answer text, not nothing or an unfinished XML/code plan."""
    t = str(text or "").strip()
    return len(t) >= 5 and not ("<Instructions>" in t or "</Code>" in t or "<EdgeList>" in t)


def mcq_options(question: str) -> Dict[int, str]:
    """Numbered options listed after the question mark."""
    tail = question.split("?", 1)[1] if "?" in question else question
    parts = re.split(r"(?:^|\s)([1-5])\.\s", " " + tail)
    return {int(parts[i]): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def options_in_answer(question: str, text: str) -> Optional[set]:
    """Distinct options named in the answer statement (after the last "answer"), or None if there is none."""
    t = str(text or "")
    marks = list(re.finditer(r"\banswer\b", t, re.I))
    if not marks:
        return None
    tail = t[marks[-1].start():]
    opts = mcq_options(question)
    # A number counts when it introduces its own option text, or follows "option" / "choice" / "is:".
    found = {k for k, v in opts.items()
             if re.search(r"(?<![\w.-])%d[.)]\s*%s" % (k, re.escape(v[:12])), tail, re.I)
             or re.search(r"\b(?:option|choice|is:?)\s*\(?%d\b(?![.,]\d)" % k, tail, re.I)}
    spans: List[Tuple[int, int]] = []
    for k, v in sorted(opts.items(), key=lambda kv: -len(kv[1])):  # a name inside a longer one does not count
        for m in (re.finditer(r"(?<!\w)" + re.escape(v) + r"(?!\w)", tail, re.I) if v else []):
            if not any(a <= m.start() and m.end() <= b for a, b in spans):
                found.add(k)
                spans.append((m.start(), m.end()))
    return found


def mcq_answer(question: str, text: str, extracted: Optional[int]) -> Tuple[Optional[int], str]:
    """(option, how it was read): the extractor's option when the response gives a single answer."""
    if not has_answer_text(text):
        return None, "no answer text"
    if extracted is None:
        return None, "extractor found no single option"
    named = options_in_answer(question, text)
    if named is not None and len(named) > 1:
        return None, f"answer names {len(named)} options"
    return int(extracted), "extractor option"


def _flatten(obj: Any) -> List[str]:
    if isinstance(obj, list):
        out: List[str] = []
        for x in obj:
            out.extend(str(v) for v in x.values()) if isinstance(x, dict) else out.append(str(x))
        return out
    return [str(obj)]


def open_ended_answer(payload: str) -> Optional[List[str]]:
    """Entity names in a raw Cypher result, or None when it holds no result."""
    try:
        obj = ast.literal_eval(payload)
    except Exception:
        try:
            obj = ast.literal_eval(payload.replace('""', '"'))
        except Exception:
            return None
    if isinstance(obj, dict):
        if not obj:
            return None
        obj = next(iter(obj.values()))
    if isinstance(obj, tuple):
        obj = obj[0] if obj else None
    if obj is None:
        return None
    return _flatten(obj)


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------

def _dataset(file: str) -> Dict[str, Any]:
    with open(os.path.join(DATASETS, file), encoding="utf-8") as fh:
        return {it["question"].strip(): it for it in json.load(fh)}


def convert(source: str) -> None:
    """Write one record per question from the raw ESCARGOT outputs in `source`."""
    os.makedirs(ARM, exist_ok=True)
    for file, btype in SETS.items():
        stem = file[:-5]
        items = _dataset(file)
        if btype == "mcq":
            with open(os.path.join(source, f"Escargot_responses_{stem}_openai.jsonl"), encoding="utf-8") as fh:
                raw = {r["question"].strip(): r for r in map(json.loads, fh)}
        else:
            with open(os.path.join(source, f"Escargot_responses_{stem}.pkl"), "rb") as fh:
                raw = {q.strip(): v for q, v in pickle.load(fh)[file].items()}
        records = []
        for q, item in items.items():
            if q not in raw:
                continue
            r = raw[q]
            rec: Dict[str, Any] = {"qid": question_id(item["question"]), "question": item["question"], "file": file,
                                   "type": btype, "expected": item["answer"], "model": "gpt-4o-mini",
                                   "agent_mode": "escargot-cypher"}
            if btype == "tf":
                answer, how = tf_answer(r["response"])
                rec.update(model_answer=answer, escargot_response=r["response"],
                           correct=answer is not None and answer == (str(item["answer"]) == "True"))
            elif btype == "mcq":
                extracted = r.get("openai_choice")
                answer, how = mcq_answer(item["question"], r["escargot_answer_text"], extracted)
                rec.update(model_answer=answer, escargot_response=r["escargot_answer_text"],
                           extractor_choice=extracted, correct=answer == int(item["answer"]))
            else:
                names = open_ended_answer(str(r["response"]))
                how = "Cypher result" if names is not None else "no result"
                rec["model_answer"] = ", ".join(names) if names is not None else None
                if names is not None:  # an unparseable result carries no metrics and scores 0
                    rec["metrics"] = compute_set_metrics(csv_to_set(item["answer"]), csv_to_set(rec["model_answer"]))
                    rec["correct"] = rec["metrics"]["exact_match"] == 1.0
                else:
                    rec["correct"] = False
            rec["read_as"] = how
            records.append(rec)
        path = os.path.join(ARM, f"{stem}__escargot-cypher__gpt-4o-mini.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{file:30s} {len(records)} records of {len(items)} questions -> {os.path.relpath(path, _REPO)}")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def report() -> Dict[str, Any]:
    """Score the records per question set and write stats/summary.json."""
    recs = load_arm([os.path.join(ARM, "*.jsonl")], label="escargot-cypher_gpt-4o-mini")
    out: Dict[str, Any] = {}
    for file, btype in SETS.items():
        keys = [k for k in recs if k[0] == file]
        if not keys:
            continue
        row: Dict[str, Any] = {"n": len(keys), "n_questions": len(_dataset(file))}
        if btype in ("tf", "mcq"):
            row["accuracy"] = 100 * sum(1 for k in keys if recs[k].get("correct")) / len(keys)
        else:
            for m in SET_METRICS:
                row[m] = 100 * (sum(1 for k in keys if recs[k].get("correct")) if m == "exact_match"
                                else sum(graded_score(recs[k], m) for k in keys)) / len(keys)
        out[file] = row
    os.makedirs(os.path.join(EXPERIMENT, "stats"), exist_ok=True)
    with open(os.path.join(EXPERIMENT, "stats", "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"note": "Percentages; open-ended answers scored with the shared scorer at load time.",
                   "sets": out}, fh, indent=2)
    for file, row in out.items():
        print(f"{file:30s} " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="raw ESCARGOT outputs -> per-question records")
    c.add_argument("--source", required=True,
                   help="directory holding the Escargot_responses_* files")
    sub.add_parser("report", help="score the records, write stats/summary.json")
    args = parser.parse_args()
    if args.cmd == "convert":
        convert(args.source)
    else:
        report()


if __name__ == "__main__":
    main()
