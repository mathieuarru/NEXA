"""LLM-only baseline: the chat model answers the benchmark questions without KG or tools.

Measures how much of the benchmark the model solves from its own knowledge.
Records follow the NEXA runner's JSONL schema, so benchmark/evaluate_benchmark.py
scores them and pairs them with other arms by question id.

The JSONL goes to benchmark/results/<experiment>/vanilla_<model>/ by default,
where <experiment> is `alzkb` for the built-in sets and the name of the dataset
folder for --input (benchmark/datasets/biohopr/... -> biohopr). Provider and
model default to BENCHMARK_PROVIDER and BENCHMARK_MODEL_NAME from .env.

Usage:
    poetry run python benchmark/baselines/baseline_vanilla.py \\
        --provider openai --model gpt-4o-mini
    poetry run python -m benchmark.evaluate_benchmark \\
        benchmark/results/alzkb/vanilla_gpt-4o-mini \\
        -o benchmark/results/alzkb/vanilla_gpt-4o-mini/report.txt
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage

from langchain_core.callbacks import get_usage_metadata_callback


############################
# Config
############################

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(BENCH_DIR)
RESULTS_DIR = os.path.join(BENCH_DIR, "results")
DATASETS_DIR = os.path.join(BENCH_DIR, "datasets", "alzkb")
SEED = 42

# Needed for `benchmark.bench_utils` when the script is run directly.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmark.bench_utils import (  # noqa: E402
    OpenEndedListAnswer,
    compute_set_metrics,
    entity_list_to_set,
    question_id,
    same_entities,
)

DATASET_SPECS: List[Tuple[str, str]] = [
    ("mcq", "MCQ_1hop.json"),
    ("mcq", "MCQ_2hop.json"),
    ("tf", "True_or_False_1hop.json"),
    ("tf", "True_or_False_2hop.json"),
    ("open_ended", "OpenEnded_1hop.json"),
    ("open_ended", "OpenEnded_2hop.json"),
]


############################
# Pydantic schemas
############################

class MCQAnswer(BaseModel):
    thinking: str = Field(description="Brief reasoning that led to the chosen option.")
    answer: int = Field(description="Selected option number (1-5).")


class TFAnswer(BaseModel):
    thinking: str = Field(description="Brief reasoning that led to the answer.")
    answer: bool = Field(description="True or False answer.")


class OpenEndedAnswer(BaseModel):
    thinking: str = Field(
        description="Brief reasoning that led to the listed entities."
    )
    answer: str = Field(
        description="Comma-separated list of entities answering the question. Example: `HTR2C,CHRM1,HTR1B,DRD4,HTR3A,SLC6A4`"
    )


############################
# System prompts
#
# They frame the task and the output format only: no KG and no external context.
############################

MCQ_PROMPT = """
You are answering a biomedical multiple-choice question. Each question lists 5 numbered options. Identify the single best answer and return its option number (1-5). If you are unsure, still return your best guess.
""".strip()

TF_PROMPT = """
You are answering a biomedical true/false question. Decide whether the statement is true or false. If you are unsure, still return your best guess.
""".strip()

OPEN_ENDED_PROMPT = """
You are answering a biomedical open-ended question that asks for a list of entities (such as drugs, genes, diseases, or pathways). Return a comma-separated list of the entities that answer the question.
""".strip()

# NEXA's OPEN_ENDED_LIST_SYSTEM_PROMPT without the retrieval instructions; the
# output contract is the same because the schema is shared.
OPEN_ENDED_LIST_PROMPT = """
You are answering a biomedical open-ended question that asks you to list entities (such as drugs, genes, diseases, or phenotypes).

Return **every** entity that answers the question, as a **list with one entity per item**.

Instructions:
- Give one entity per list item. Many biomedical names contain commas (e.g. 'dermatitis herpetiformis, familial'): keep such a name whole inside a single item, never split it across items.
- Use the entity's standard name.
- If you are unsure, still return your best guess rather than an empty list.
""".strip()


BENCH_TYPES: Dict[str, Dict[str, Any]] = {
    "mcq": {"schema": MCQAnswer, "prompt": MCQ_PROMPT},
    "tf": {"schema": TFAnswer, "prompt": TF_PROMPT},
    "open_ended": {"schema": OpenEndedAnswer, "prompt": OPEN_ENDED_PROMPT},
    # Imported, not copied: its docstring and Field descriptions are sent to the
    # LLM, so it must be the exact schema NEXA is scored with.
    "open_ended_list": {"schema": OpenEndedListAnswer, "prompt": OPEN_ENDED_LIST_PROMPT},
}


############################
# Scoring helpers (mirror benchmark/benchmark.py)
############################

def csv_to_set(raw: Any) -> Set[str]:
    """Set of stripped names from a comma-separated string."""
    if not raw:
        return set()
    if not isinstance(raw, str):
        raw = str(raw)
    return {t.strip() for t in raw.split(",") if t and t.strip()}


def list_to_set(raw: Any) -> Set[str]:
    """Set of stripped names from a list, or from a comma-separated string."""
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple, set)):
        return {str(t).strip() for t in raw if str(t).strip()}
    return csv_to_set(raw)


# List answers are scored with the runner's own helpers (entity_list_to_set,
# compute_set_metrics, same_entities), imported so scoring cannot diverge.


def evaluate_correctness(bench_type: str, expected: Any, model_answer: Any) -> bool:
    """Whether `model_answer` is correct for a question of type `bench_type`."""
    if bench_type == "open_ended_list":
        return same_entities(entity_list_to_set(expected), entity_list_to_set(model_answer))
    if bench_type == "open_ended":
        return same_entities(csv_to_set(expected), list_to_set(model_answer))
    if bench_type == "mcq":
        try:
            exp = int(expected)
        except Exception:
            return False
        return isinstance(model_answer, int) and model_answer == exp
    if bench_type == "tf":
        s = str(expected).strip().lower()
        if s not in {"true", "false"}:
            return False
        exp = s == "true"
        return isinstance(model_answer, bool) and model_answer == exp
    return False


############################
# LLM
############################

def build_llm(provider: str, model: str):
    """Chat model for `provider` and `model`, at temperature 0 where allowed."""
    p = provider.strip().lower()
    if p in {"openai", "azure-openai"}:
        from langchain_openai import ChatOpenAI
        # Reasoning models (o-series, gpt-5) only accept the default temperature 1.0.
        temperature = 1.0 if model.startswith(("o1", "o3", "o4", "gpt-5")) else 0.0
        return ChatOpenAI(model=model, temperature=temperature)
    if p in {"anthropic", "claude"}:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=0)
    if p in {"google", "gemini"}:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=0)
    if p == "ollama":
        from langchain_ollama import ChatOllama
        from nexa.config import params
        from nexa.utils.token_utils import get_context_window
        # Same context and output budget as the NEXA agents: Ollama otherwise
        # truncates the prompt at 4096 tokens and leaves the output unbounded.
        return ChatOllama(
            model=model,
            temperature=0,
            base_url=params.OLLAMA_BASE_URL,
            num_ctx=get_context_window(model),
            num_predict=params.NEXA_LLM_MAX_OUTPUT_TOKENS,
        )
    raise ValueError(f"Unsupported provider: {provider}")


def aggregate_usage(usage_metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, int]]:
    """Sum LangChain UsageMetadata entries (keyed by model) into one dict."""
    if not usage_metadata:
        return None
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for usage in usage_metadata.values():
        if not isinstance(usage, dict):
            continue
        for key in totals:
            try:
                totals[key] += int(usage.get(key) or 0)
            except (TypeError, ValueError):
                pass
    return totals if any(totals.values()) else None


############################
# Per-question runner
############################

def run_one_question(llm, bench_type: str, fname: str, question: str, expected: Any, model: str = "") -> Dict[str, Any]:
    """Ask the model one question and return its JSONL record."""
    cfg = BENCH_TYPES[bench_type]
    schema = cfg["schema"]
    sys_prompt = cfg["prompt"]
    structured_llm = llm.with_structured_output(schema)

    t0 = time.perf_counter()
    model_answer: Any = "NO_RESPONSE"
    thinking = ""
    error: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    try:
        with get_usage_metadata_callback() as cb:
            result = structured_llm.invoke([
                SystemMessage(content=sys_prompt),
                HumanMessage(content=question),
            ])
        usage = aggregate_usage(cb.usage_metadata)
        thinking = getattr(result, "thinking", "") or ""
        model_answer = result.answer
        if bench_type == "open_ended":
            model_answer = list(csv_to_set(model_answer))
        elif bench_type == "open_ended_list":
            # No stripping: a trailing space encodes PrimeKG's "(disease)"
            # marker in BioHopR gold answers.
            model_answer = [str(x) for x in (model_answer or []) if str(x)]
    except Exception as e:
        error = f"llm_error: {e}"
    latency_ms = int((time.perf_counter() - t0) * 1000)

    correct = False
    if error is None:
        correct = evaluate_correctness(bench_type, expected, model_answer)

    record: Dict[str, Any] = {
        # The scorer pairs arms question by question on (type, qid).
        "qid": question_id(question),
        "type": bench_type,
        "question": question,
        "expected": expected,
        "model_answer": model_answer,
        "thinking": thinking,
        "latency_ms": latency_ms,
        "correct": correct,
        "file": f"{fname}__vanilla",
        "dataset": fname,
        "mode": "vanilla",
        "model": model,
        "agent_mode": "vanilla",
    }
    if usage:
        record["usage"] = usage
    if bench_type == "open_ended" and error is None:
        record["metrics"] = compute_set_metrics(csv_to_set(expected), list_to_set(model_answer))
    elif bench_type == "open_ended_list" and error is None:
        record["metrics"] = compute_set_metrics(
            entity_list_to_set(expected), entity_list_to_set(model_answer)
        )
    if error:
        record["error"] = error
    return record


############################
# Dataset loop
############################

def _sample_dataset(path: str, n: Optional[int], rng: random.Random) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if n is None or len(data) <= n:
        return data
    return rng.sample(data, n)


def write_result_line(jsonl_path: str, record: Dict[str, Any]):
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_questions(jsonl_path: str) -> Set[str]:
    """Return the set of question texts already recorded in an existing JSONL."""
    seen: Set[str] = set()
    if not os.path.isfile(jsonl_path):
        return seen
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = obj.get("question")
            if isinstance(q, str):
                seen.add(q)
    return seen


def default_arm_dir(system: str, model: str, input_path: Optional[str]) -> str:
    """Default output directory: benchmark/results/<experiment>/<system>_<model>/.

    <experiment> is the folder holding the question set (`alzkb` for the
    built-in sets). ':' is replaced because Windows forbids it in paths.
    """
    experiment = "alzkb"
    if input_path:
        parent = os.path.basename(os.path.dirname(os.path.abspath(input_path)))
        experiment = parent if parent and parent != "datasets" else experiment
    return os.path.join(RESULTS_DIR, experiment, f"{system}_{model.replace(':', '-').replace('/', '_')}")


def main():
    load_dotenv(os.path.join(REPO_ROOT, ".env"))

    parser = argparse.ArgumentParser(description="Vanilla LLM baseline (no KG) for NEXA comparison.")
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=None,
        help="Limit to N random questions per dataset (seed=42). Omit to run on the full dataset.",
    )
    parser.add_argument("--provider", default=os.getenv("BENCHMARK_PROVIDER", "").strip() or "openai")
    parser.add_argument("--model", default=os.getenv("BENCHMARK_MODEL_NAME", "").strip() or "gpt-4o-mini")
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated subset of dataset filenames (e.g. MCQ_1hop.json,True_or_False_1hop.json) or 'all'.",
    )
    parser.add_argument(
        "--input",
        help="Path to an arbitrary benchmark JSON file (overrides --datasets). Requires --type.",
    )
    parser.add_argument(
        "--type",
        choices=["mcq", "tf", "open_ended", "open_ended_list"],
        help="Benchmark type of --input.",
    )
    parser.add_argument("--out-suffix", default="", help="Optional suffix appended to the JSONL filename.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write the JSONL into. Default: the arm directory "
             "benchmark/results/<experiment>/vanilla_<model>/ (see the module docstring).",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to an existing JSONL to continue. Already-processed question texts will be skipped and new records appended.",
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(args.out_dir) if args.out_dir else default_arm_dir("vanilla", args.model, args.input)
    os.makedirs(results_dir, exist_ok=True)

    llm = build_llm(args.provider, args.model)

    if args.input:
        if not args.type:
            raise SystemExit("--input requires --type {mcq,tf,open_ended,open_ended_list}")
        specs = [(args.type, args.input)]
    elif args.datasets.strip().lower() == "all":
        specs = DATASET_SPECS
    else:
        wanted = {x.strip() for x in args.datasets.split(",") if x.strip()}
        specs = [s for s in DATASET_SPECS if s[1] in wanted]
        if not specs:
            raise SystemExit(f"No datasets matched --datasets={args.datasets}")

    rng = random.Random(SEED)
    if args.resume:
        out_path = os.path.abspath(args.resume)
        if not os.path.isfile(out_path):
            raise SystemExit(f"--resume path does not exist: {out_path}")
        already = load_existing_questions(out_path)
        print(f"[vanilla] resuming from {out_path} ({len(already)} questions already done)")
    else:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = args.model.replace("/", "_").replace(":", "_")
        suffix = f"_{args.out_suffix}" if args.out_suffix else ""
        out_path = os.path.join(results_dir, f"baseline_vanilla_{model_slug}{suffix}_{ts}.jsonl")
        already = set()

    print(f"[vanilla] provider={args.provider} model={args.model}")
    print(f"[vanilla] writing to {out_path}")

    total = 0
    correct = 0
    for bench_type, fname in specs:
        path = fname if os.path.isabs(fname) or os.path.isfile(fname) else os.path.join(DATASETS_DIR, fname)
        label = os.path.basename(fname)
        if not os.path.isfile(path):
            print(f"[vanilla] skip missing {path}")
            continue
        items = _sample_dataset(path, args.samples_per_dataset, rng)
        for i, item in enumerate(items, 1):
            question = item["question"]
            expected = item.get("answer", "")
            if question in already:
                continue
            record = run_one_question(llm, bench_type, label, question, expected, model=args.model)
            write_result_line(out_path, record)
            already.add(question)
            total += 1
            if record.get("correct"):
                correct += 1
            status = "OK" if record.get("correct") else ("ERR" if "error" in record else "X")
            print(f"[vanilla] {label} {i}/{len(items)} {status} latency={record['latency_ms']}ms")

    print(f"[vanilla] done. {correct}/{total} correct. JSONL: {out_path}")


if __name__ == "__main__":
    main()
