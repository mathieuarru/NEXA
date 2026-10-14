"""GraphCypherQAChain baseline (langchain-neo4j) for comparison with NEXA.

Runs the benchmark question sets through GraphCypherQAChain and writes JSONL
records in the NEXA runner's schema, so benchmark/evaluate_benchmark.py scores
both and pairs them by question id. No NEXA agent code is used: only the shared
scoring helpers in benchmark.bench_utils and, for Ollama, NEXA's token budget.

GraphCypherQAChain passes at most `top_k` Cypher rows to its answering LLM
(library default 10, which caps list answers at 10 entities). `--top-k`
(default 10000) raises that cap; the value is recorded in each record.
`allow_dangerous_requests=True` is required because the LLM writes arbitrary
Cypher; the benchmark graphs are local.

The JSONL goes to benchmark/results/<experiment>/graphcypher_<model>/ by
default, where <experiment> is `alzkb` for the built-in sets and the name of
the dataset folder for --input.

Usage:
    poetry install --with benchmark
    poetry run python benchmark/baselines/baseline_graphcypher.py \\
        --profile "Alzkb 121" --samples-per-dataset 5
    poetry run python -m benchmark.evaluate_benchmark \\
        benchmark/results/alzkb/graphcypher_gpt-4o-mini \\
        -o benchmark/results/alzkb/graphcypher_gpt-4o-mini/report.txt
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
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph

from langchain_core.callbacks import get_usage_metadata_callback


############################
# Config
############################

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(BENCH_DIR)
RESULTS_DIR = os.path.join(BENCH_DIR, "results")
DATASETS_DIR = os.path.join(BENCH_DIR, "datasets", "alzkb")
PROFILES_PATH = os.path.join(REPO_ROOT, "api_data", "kg_profiles.json")
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
# Pydantic answer schemas
############################

class MCQAnswer(BaseModel):
    thinking: str = Field(description="Reasoning based on results.")
    answer: int = Field(description="Selected option number.")


class TFAnswer(BaseModel):
    thinking: str = Field(description="Reasoning based on results.")
    answer: bool = Field(description="True or False answer.")


class OpenEndedAnswer(BaseModel):
    thinking: str = Field(
        description="A detailed explanation of the reasoning used to arrive at the answer, based on the provided results."
    )
    answer: str = Field(
        description="A comma-separated list of entities that answer the question, based on the results. Example: `HTR2C,CHRM1,HTR1B,DRD4,HTR3A,SLC6A4`"
    )


############################
# Extraction prompts
#
# GraphCypherQAChain does the reasoning. This second LLM call only parses the
# chain's text answer into the scored format (int / bool / entity list); it
# must not reason about the question or override the chain's answer.
############################

MCQ_EXTRACTION_PROMPT = """
You are a strict format extractor. You receive a multiple-choice question and a free-text answer produced by another system. Your only job is to identify which option number (1-5) the answer corresponds to and return it. Do not re-evaluate the question. Do not consult any external knowledge. If the answer text is empty, contradictory, or does not name any of the options, return 0.
""".strip()

TF_EXTRACTION_PROMPT = """
You are a strict format extractor. You receive a true/false question and a free-text answer produced by another system. Your only job is to extract whether the answer asserts True or False, and return that boolean. Do not re-evaluate the question. Do not consult any external knowledge. If the answer text is ambiguous or empty, return False.
""".strip()

OPEN_ENDED_EXTRACTION_PROMPT = """
You are a strict format extractor. You receive a question asking for a list of entities and a free-text answer produced by another system. Your only job is to extract that list and return it as a comma-separated string of entity names, preserving the names exactly as they appear in the answer. Do not add entities. Do not consult any external knowledge. If no entities are listed, return an empty string.
""".strip()


OPEN_ENDED_LIST_EXTRACTION_PROMPT = """
You are a strict format extractor. You receive a question asking for a list of entities and a free-text answer produced by another system. Your only job is to extract that list and return it with ONE ENTITY PER LIST ITEM, preserving each name exactly as it appears in the answer. Many biomedical names contain a comma (e.g. 'dermatitis herpetiformis, familial'): such a name is ONE entity and must stay whole inside a single item. Do not add entities. Do not drop entities. Do not consult any external knowledge. If no entities are listed, return an empty list.
""".strip()


BENCH_TYPES: Dict[str, Dict[str, Any]] = {
    "mcq": {"schema": MCQAnswer, "prompt": MCQ_EXTRACTION_PROMPT},
    "tf": {"schema": TFAnswer, "prompt": TF_EXTRACTION_PROMPT},
    "open_ended": {"schema": OpenEndedAnswer, "prompt": OPEN_ENDED_EXTRACTION_PROMPT},
    # Imported, not copied: its docstring and Field descriptions are sent to the
    # LLM, so it must be the exact schema NEXA is scored with.
    "open_ended_list": {"schema": OpenEndedListAnswer, "prompt": OPEN_ENDED_LIST_EXTRACTION_PROMPT},
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
# LLM + KG setup
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
    """Sum LangChain usage metadata (keyed by model) into one dict.

    Covers the chain's Cypher and QA calls as well as the parsing call.
    """
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


def load_profile(profile_name: str) -> Dict[str, str]:
    """Connection settings of a KG profile from api_data/kg_profiles.json."""
    with open(PROFILES_PATH, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    if profile_name not in profiles:
        raise KeyError(f"Profile '{profile_name}' not found in {PROFILES_PATH}. Available: {list(profiles)}")
    return profiles[profile_name]


def build_chain(profile: Dict[str, str], llm, top_k: int) -> Tuple[GraphCypherQAChain, Neo4jGraph]:
    """GraphCypherQAChain over the profile's Neo4j database."""
    graph = Neo4jGraph(
        url=f"bolt://{profile['host']}:{profile['port']}",
        username=profile["username"],
        password=profile["password"],
        database=profile.get("db_name", "neo4j"),
    )
    graph.refresh_schema()
    # validate_cypher: check relationship patterns of the generated Cypher
    #   against the schema (fixes reversed directions).
    # use_function_response: pass KG results to the QA step as a tool message.
    # top_k: rows passed to the QA LLM (library default 10); forwarded to the
    #   chain through **kwargs.
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=False,
        return_intermediate_steps=True,
        allow_dangerous_requests=True,
        validate_cypher=True,
        use_function_response=True,
        top_k=top_k,
    )
    return chain, graph


############################
# Per-question runner
############################

def _extract_chain_trace(chain_out: Dict[str, Any]) -> Tuple[str, Any, str]:
    """(Cypher query, KG context, text answer) from GraphCypherQAChain output."""
    cypher = ""
    context: Any = []
    steps = chain_out.get("intermediate_steps") or []
    for step in steps:
        if not isinstance(step, dict):
            continue
        if "query" in step and not cypher:
            cypher = step["query"] or ""
        if "context" in step:
            context = step["context"]
    text_answer = chain_out.get("result", "") or ""
    return cypher, context, text_answer


def parse_with_schema(llm, bench_type: str, question: str, text_answer: str) -> Any:
    """Parse the chain's text answer into the answer format of `bench_type`.

    The parser sees only the question and the text answer, not the KG context.
    """
    cfg = BENCH_TYPES[bench_type]
    schema = cfg["schema"]
    sys_prompt = cfg["prompt"]
    structured_llm = llm.with_structured_output(schema)

    user_payload = (
        f"Question:\n{question}\n\n"
        f"Answer text to parse:\n{text_answer}"
    )
    result = structured_llm.invoke([
        SystemMessage(content=sys_prompt),
        HumanMessage(content=user_payload),
    ])
    answer = result.answer
    if bench_type == "open_ended":
        answer = list(csv_to_set(answer))
    elif bench_type == "open_ended_list":
        # No stripping: a trailing space encodes PrimeKG's "(disease)" marker
        # in BioHopR gold answers.
        answer = [str(x) for x in (answer or []) if str(x)]
    return answer


def run_one_question(chain, llm, bench_type: str, fname: str, question: str, expected: Any, top_k: int, model: str = "") -> Dict[str, Any]:
    """Run one question through the chain and return its JSONL record."""
    t0 = time.perf_counter()
    cypher = ""
    text_answer = ""
    model_answer: Any = "NO_RESPONSE"
    thinking = ""
    error: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    def _run():
        nonlocal cypher, text_answer, thinking, model_answer, error
        try:
            chain_out = chain.invoke({"query": question})
            cypher, _, text_answer = _extract_chain_trace(chain_out)
            # The chain's text answer is its reasoning; record it as `thinking`.
            thinking = text_answer
            try:
                model_answer = parse_with_schema(llm, bench_type, question, text_answer)
            except Exception as parse_err:
                error = f"parse_error: {parse_err}"
        except Exception as e:
            error = f"chain_error: {e}"

    with get_usage_metadata_callback() as cb:
        _run()
    usage = aggregate_usage(cb.usage_metadata)

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
        "cypher_first_attempt": cypher,
        "cypher_final": cypher,
        "n_correction_iterations": 0,
        "first_attempt_success": (error is None and bool(cypher)),
        "correct": correct,
        "file": f"{fname}__graphcypher",
        "dataset": fname,
        "mode": "graphcypher",
        "model": model,
        "agent_mode": "graphcypher",
        "top_k": top_k,
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

    parser = argparse.ArgumentParser(description="GraphCypherQAChain baseline benchmark for NEXA comparison.")
    parser.add_argument("--profile", default="Alzkb 121", help="KG profile name (must exist in api_data/kg_profiles.json)")
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
        "--top-k",
        type=int,
        default=10000,
        help="Max Cypher result rows handed to the QA LLM (GraphCypherQAChain.top_k). "
             "The library default is 10, which caps open-ended answers at 10 entities; "
             "raise it for a fair comparison on list-style questions. Very large sets may "
             "still overflow the QA LLM context window (a real model limit).",
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
             "benchmark/results/<experiment>/graphcypher_<model>/ (see the module docstring).",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to an existing JSONL to continue. Already-processed question texts will be skipped and new records appended to the same file.",
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(args.out_dir) if args.out_dir else default_arm_dir("graphcypher", args.model, args.input)
    os.makedirs(results_dir, exist_ok=True)

    profile = load_profile(args.profile)
    llm = build_llm(args.provider, args.model)
    chain, graph = build_chain(profile, llm, args.top_k)

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
        print(f"[baseline] resuming from {out_path} ({len(already)} questions already done)")
    else:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        profile_slug = args.profile.replace(" ", "_")
        model_slug = args.model.replace("/", "_").replace(":", "-")
        suffix = f"_{args.out_suffix}" if args.out_suffix else ""
        out_path = os.path.join(results_dir, f"baseline_graphcypher_{model_slug}_{profile_slug}{suffix}_{ts}.jsonl")
        already = set()

    print(f"[baseline] provider={args.provider} model={args.model} profile={args.profile} top_k={args.top_k}")
    print(f"[baseline] writing to {out_path}")

    total = 0
    correct = 0
    for bench_type, fname in specs:
        path = fname if os.path.isabs(fname) or os.path.isfile(fname) else os.path.join(DATASETS_DIR, fname)
        label = os.path.basename(fname)
        if not os.path.isfile(path):
            print(f"[baseline] skip missing {path}")
            continue
        items = _sample_dataset(path, args.samples_per_dataset, rng)
        for i, item in enumerate(items, 1):
            question = item["question"]
            expected = item.get("answer", "")
            if question in already:
                continue
            record = run_one_question(chain, llm, bench_type, label, question, expected, args.top_k, model=args.model)
            write_result_line(out_path, record)
            already.add(question)
            total += 1
            if record.get("correct"):
                correct += 1
            status = "OK" if record.get("correct") else ("ERR" if "error" in record else "X")
            print(f"[baseline] {label} {i}/{len(items)} {status} latency={record['latency_ms']}ms")

    print(f"[baseline] done. {correct}/{total} correct. JSONL: {out_path}")


if __name__ == "__main__":
    main()
