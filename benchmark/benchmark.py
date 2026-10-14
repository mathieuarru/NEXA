"""NEXA benchmark runner for both agents.

  --agent-mode direct   Cypher generation and correction pipeline (NexaAgent)
  --agent-mode flex     LangGraph ReAct agent with granular tools (NexaFlex)

Each question yields one JSONL record: answer, correctness, telemetry
(latency, tokens, Cypher attempts, Flex tool trace) and identity fields
(`qid`, `hops`, `model`, `run_tag`) used to pair runs. Each output file gets a
`<name>.jsonl.meta.json` manifest (model, config, env flags, git commit, timestamps).

Examples:
  python -m benchmark.benchmark --run-all --profile "Alzkb 121" \\
      --out-subdir alzkb/nexa-direct_gpt-4o-mini
  BENCHMARK_MODEL_NAME=gpt-5-mini python -m benchmark.benchmark \\
      --agent-mode flex --run-all --profile "Alzkb 121" --run-tag run1 \\
      --out-subdir alzkb/nexa-flex_gpt-5-mini
  python -m benchmark.benchmark --type mcq --input benchmark/datasets/alzkb/MCQ_1hop.json \\
      --profile "Alzkb 121" --retry-errors
"""

import argparse
import glob
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_core.callbacks import get_usage_metadata_callback
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from pydantic import BaseModel, Field

from nexa.api.profile_manager import (
    get_profile_with_password,
    load_descriptions_from_profile,
    load_dynamic_examples_for_profile,
    load_identifying_properties_from_description,
    load_identifying_properties_for_profile,
)
from nexa.flex import NexaFlex
from nexa.config import params
from nexa.core.nexa_agent import NexaAgent, get_llm
from nexa.knowledge_graph.neo4j_handler import Neo4jHandler
from nexa.models.dynamic_examples import build_and_store_examples
from nexa.result_store import ResultStore

try:
    from benchmark.bench_utils import (
        append_jsonl,
        compute_set_metrics,
        csv_to_set,
        ensure_dir,
        entity_list_to_set,
        OpenEndedListAnswer,
        git_commit_short,
        gold_covered,
        hops_from_name,
        list_to_set,
        question_id,
        read_jsonl,
        safe_filename,
        same_entities,
        write_jsonl,
    )
except ImportError:  # executed as a plain script from inside benchmark/
    from bench_utils import (
        append_jsonl,
        compute_set_metrics,
        csv_to_set,
        ensure_dir,
        entity_list_to_set,
        OpenEndedListAnswer,
        git_commit_short,
        gold_covered,
        hops_from_name,
        list_to_set,
        question_id,
        read_jsonl,
        safe_filename,
        same_entities,
        write_jsonl,
    )


############################
# Config
############################

MODEL_NAME = params.BENCHMARK_MODEL_NAME
BENCHMARK_PROVIDER = params.BENCHMARK_PROVIDER

RESULTS_DIR = os.path.join(os.path.dirname(__file__) or ".", "results")
# --run-all runs every AlzKB question set; any other set is passed with --input.
DATASETS_DIR = os.path.join(os.path.dirname(__file__) or ".", "datasets", "alzkb")


@dataclass
class RunConfig:
    """Everything that parameterizes one benchmark run (recorded in the manifest)."""

    agent_mode: str = "direct"               # direct | flex
    kg_mode: str = "Force"                   # Off | Auto | Force (direct mode only)
    profile_name: Optional[str] = None
    examples_mode: Optional[str] = None      # direct mode only
    generation_mode: str = "one_shot_fallback"  # direct mode only
    recursion_limit: int = 150               # flex mode only
    checkpointer: str = "memory"             # flex mode only: memory | postgres
    limit: Optional[int] = None
    overwrite: bool = False
    retry_errors: bool = False
    retry_all_errors: bool = False
    max_retries: int = 2                     # retries on raised exceptions (transient API errors)
    run_tag: Optional[str] = None            # disambiguates repeated runs (run1/run2/...)
    out_subdir: Optional[str] = None         # nest outputs under results/<out_subdir>/


############################
# Pydantic answer schemas
############################

class MCQAnswer(BaseModel):
    thinking: str = Field(description="Reasoning based on KG results.")
    answer: str = Field(
        description=(
            "The FULL TEXT of the selected option, copied verbatim from the "
            "question's option list (e.g. for '1. CANX 2. NR2F6 3. MAT2B', "
            "return 'CANX'"
        )
    )


class TFAnswer(BaseModel):
    thinking: str = Field(description="Reasoning based on KG results.")
    answer: bool = Field(description="True or False answer.")


class OpenEndedAnswer(BaseModel):
    """Structured format for Open-Ended Question answers."""
    thinking: str = Field(description="A detailed explanation of the reasoning used to arrive at the answer, based on the provided knowledge graph data.")
    answer: str = Field(description="A comma-separated list of entities that answer the question, based on the knowledge graph data. Example: `HTR2C,CHRM1,HTR1B,DRD4,HTR3A,SLC6A4`")


############################
# System prompts
############################

MCQ_SYSTEM_PROMPT = """
You are a biomedical expert with access to a knowledge graph (KG).
You will be asked multiple choice questions (MCQs), each offering 5 numbered options.
Your task is to identify the single correct answer using only the provided KG results.
Instructions:
- Carefully analyze the Cypher query to understand what it retrieves.
- Treat the KG results as complete and exhaustive (closed world).
- If an entity is not present in the result, it does not match.
- Return the FULL TEXT of the selected option in the `answer` field, copied verbatim
  from the question (e.g. for "1. CANX 2. NR2F6 3. MAT2B", return "CANX" — not "1").
  Do NOT include the option number or any prefix.
""".strip()


TF_SYSTEM_PROMPT = """
You are a biomedical expert with access to a knowledge graph (KG).
You will be asked a true/false question.
Your task is to determine if the statement is True or False using only the provided KG results.
Instructions:
- Carefully analyze the Cypher query to understand what it retrieves.
- Treat the KG results as complete and exhaustive (closed world).
- If an entity is not present in the result, it does not match.
""".strip()


OPEN_ENDED_SYSTEM_PROMPT = """
You are a biomedical expert with access to a knowledge graph (KG).

You will be asked an open-ended question, likely asking you to list biomedical entities (such as drugs, genes, or diseases).

Your task is to provide the answer as a **comma-separated list** of entities, using **only** the provided results from the knowledge graph.

Instructions:
- Carefully analyze the Cypher query to understand what it retrieves.
- **If the question implies an INTERSECTION, AND condition** (e.g., "entities related to A **and** B"), return **only the entities that are connected to both A and B** (i.e., the intersection).
- **If the question implies an UNION, OR condition** (e.g., "entities related to A **or** B"), return **all entities connected to either A or B** (i.e., the union).
- Treat the results from the knowledge graph as **complete and exhaustive**.
""".strip()


OPEN_ENDED_LIST_SYSTEM_PROMPT = """
You are a biomedical expert with access to a knowledge graph (KG).

You will be asked an open-ended question asking you to list biomedical entities (such as drugs, genes, diseases, or phenotypes).

Your task is to return **every** entity that answers the question, using **only** the provided results from the knowledge graph, as a **list with one entity per item**.

Instructions:
- Carefully analyze the Cypher query to understand what it retrieves.
- **Copy each entity name verbatim** from the knowledge graph results. Do not reword, expand, abbreviate or re-case it.
- **One entity per list item.** Many biomedical names contain a comma (e.g. "dermatitis herpetiformis, familial", "chromosome 17q11.2 deletion syndrome, 1.4Mb"). Such a name is a SINGLE entity and must stay whole inside one item. Never split it, and never merge two entities into one item.
- **If the question implies an INTERSECTION, AND condition** (e.g., "entities related to A **and** B"), return **only the entities that are connected to both A and B** (i.e., the intersection).
- **If the question implies an UNION, OR condition** (e.g., "entities related to A **or** B"), return **all entities connected to either A or B** (i.e., the union).
- The question asks for **all** matching entities. Answers are frequently long — hundreds of entities. **List every one of them**; do not stop early, do not summarize, do not write "and others".
- Treat the results from the knowledge graph as **complete and exhaustive**.
""".strip()


############################
# MCQ option parsing (answer text to option number)
############################

_MCQ_OPTION_SPLIT_RE = re.compile(r'(?:^|\s)([1-5])\.\s+')
_MCQ_LEADING_NUMBER_RE = re.compile(r'^\s*(?:option\s+)?([1-5])(?:[\.\):]|\s|$)', re.I)
_MCQ_NORMALIZE_RE = re.compile(r'[^a-z0-9]+')


def _normalize(s: str) -> str:
    return _MCQ_NORMALIZE_RE.sub(' ', s.lower()).strip()


def parse_mcq_options(question: str) -> Dict[int, str]:
    """Extract the numbered options (1-5) from an MCQ question text.

    A split needs a digit 1-5 followed by a dot and whitespace, so numbers inside
    option text ("Alzheimer Disease, Familial, 3, with...") do not split it.
    """
    matches = list(_MCQ_OPTION_SPLIT_RE.finditer(question))
    if len(matches) < 2:
        return {}
    options: Dict[int, str] = {}
    for i, m in enumerate(matches):
        num = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(question)
        options[num] = question[start:end].strip()
    return options


def match_mcq_string_to_option(answer_str: str, options: Dict[int, str]) -> Optional[int]:
    """Map an LLM free-text answer to one of the option numbers 1-5.

    Tries a leading number ("1", "Option 3", "3."), then an exact normalized
    match, then the best fuzzy match (SequenceMatcher ratio >= 0.5).
    Returns None when nothing matches; the caller maps that to 0 (invalid).
    """
    if not answer_str or not options:
        return None
    s = str(answer_str).strip()

    m = _MCQ_LEADING_NUMBER_RE.match(s)
    if m:
        n = int(m.group(1))
        if n in options:
            return n

    s_norm = _normalize(s)
    if not s_norm:
        return None

    best_n, best_score = None, 0.0
    for n, opt in options.items():
        opt_norm = _normalize(opt)
        if s_norm == opt_norm:
            return n
        score = SequenceMatcher(None, s_norm, opt_norm).ratio()
        if score > best_score:
            best_score = score
            best_n = n
    return best_n if best_score >= 0.5 else None


############################
# Type registry
############################

BENCH_TYPES = {
    "mcq": {
        "schema": MCQAnswer,
        "prompt": MCQ_SYSTEM_PROMPT,
        "prefix": "MCQ_",
    },
    "tf": {
        "schema": TFAnswer,
        "prompt": TF_SYSTEM_PROMPT,
        "prefix": "True_or_False_",
    },
    "open_ended": {
        "schema": OpenEndedAnswer,
        "prompt": OPEN_ENDED_SYSTEM_PROMPT,
        "prefix": "OpenEnded_",
    },
    # Open-ended with a JSON-list gold (BioHopR). Its own prefix keeps --run-all
    # from mixing it with the CSV-gold OpenEnded_* sets.
    "open_ended_list": {
        "schema": OpenEndedListAnswer,
        "prompt": OPEN_ENDED_LIST_SYSTEM_PROMPT,
        "prefix": "BioHopR_",
    },
}


############################
# Answer extraction + correctness
############################

def extract_model_answer(
    bench_type: str, ans: Any, question: str
) -> Tuple[Any, Optional[str], Optional[str]]:
    """Normalize the raw `answer` payload into its evaluable form.

    Returns (model_answer, model_answer_text, error). `model_answer_text` is set
    for MCQ only, to audit the text -> option mapping.
    """
    if bench_type == "open_ended":
        if isinstance(ans, list):
            return [str(x) for x in ans], None, None
        if isinstance(ans, str):
            return ans, None, None
        return ans, None, f"Invalid OpenEnded answer type: expected list or str, got {type(ans).__name__}"

    if bench_type == "open_ended_list":
        if isinstance(ans, list):
            return [str(x) for x in ans], None, None
        # A single string is scored as one entity, never split on commas.
        if isinstance(ans, str):
            return [ans], None, None
        return ans, None, f"Invalid OpenEndedList answer type: expected list, got {type(ans).__name__}"

    if bench_type == "mcq":
        text = ans if isinstance(ans, str) else (str(ans) if ans is not None else "")
        mapped = match_mcq_string_to_option(text, parse_mcq_options(question))
        return (mapped if mapped is not None else 0), text, None

    if bench_type == "tf":
        if isinstance(ans, bool):
            return ans, None, None
        return ans, None, f"Invalid TF answer type: expected bool, got {type(ans).__name__}"

    return ans, None, f"Unknown bench type: {bench_type}"


def evaluate_correctness(bench_type: str, expected: Any, model_answer: Any) -> bool:
    """Whether the answer is correct (open-ended: exact set match under entity_key)."""
    if bench_type == "open_ended":
        return same_entities(csv_to_set(expected), list_to_set(model_answer))

    if bench_type == "open_ended_list":
        # The gold is a JSON array: parse both sides as lists, never as CSV.
        return same_entities(entity_list_to_set(expected), entity_list_to_set(model_answer))

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
        return isinstance(model_answer, bool) and model_answer == (s == "true")

    return False


############################
# Usage (token) accounting
############################

def aggregate_usage(usage_metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, int]]:
    """Sum LangChain UsageMetadata entries (keyed by model name) into one dict."""
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
# Resume / retry bookkeeping
############################

def needs_rerun(record: Dict[str, Any], retry_all: bool) -> bool:
    """Whether an errored record should be re-run.

    Only infrastructure failures (`error_source == "exception"`: API outage,
    rate limit, network) are retried. The agent's own failures (recursion
    limit, no or invalid answer) count as incorrect: re-running only those
    would bias accuracy upward. `retry_all` retries everything (debugging only).
    """
    if not record.get("error") and record.get("model_answer") != "NO_RESPONSE":
        return False  # clean record
    if retry_all:
        return True
    return record.get("error_source") == "exception"


def prepare_output_file(
    out_path: str, overwrite: bool, retry_errors: bool, retry_all_errors: bool
) -> Set[str]:
    """Set up the output file for this run; return the question texts to skip.

    overwrite: truncate and redo everything. retry_errors: drop the records
    `needs_rerun` selects and redo them. Default: resume, skipping every
    question already present.
    """
    if overwrite:
        with open(out_path, "w", encoding="utf-8"):
            pass
        return set()

    existing = read_jsonl(out_path)
    if retry_errors or retry_all_errors:
        kept = [r for r in existing if not needs_rerun(r, retry_all_errors)]
        if len(kept) != len(existing):
            write_jsonl(out_path, kept)
            print(f"  retry-errors: kept {len(kept)} records, re-running {len(existing) - len(kept)}")
        return {r.get("question") for r in kept if isinstance(r.get("question"), str)}

    return {r.get("question") for r in existing if isinstance(r.get("question"), str)}


def manifest_path(out_path: str) -> str:
    return out_path + ".meta.json"


def write_manifest(out_path: str, cfg: RunConfig, dataset_file: str, extra: Dict[str, Any]) -> None:
    """Write the run manifest `<out_path>.meta.json`."""
    payload = {
        "dataset_file": dataset_file,
        "model": MODEL_NAME,
        "provider": BENCHMARK_PROVIDER,
        "git_commit": git_commit_short(),
        "config": asdict(cfg),
        # Env-driven behaviour flags, so each ablation arm's files say which variant produced them.
        "flags": {
            "NEXA_COMPLETENESS_GUARD": params.NEXA_COMPLETENESS_GUARD,
            "NEXA_NO_LIMIT_PROMPT": params.NEXA_NO_LIMIT_PROMPT,
            "NEXA_FLEX_ANSWER_RESCUE": params.NEXA_FLEX_ANSWER_RESCUE,
            "NEXA_PROPERTY_MATCHING": params.NEXA_PROPERTY_MATCHING,
            "NEXA_LLM_MAX_OUTPUT_TOKENS": params.NEXA_LLM_MAX_OUTPUT_TOKENS,
            "NEXA_LLM_MAX_PROPERTY_TOKENS": params.NEXA_LLM_MAX_PROPERTY_TOKENS,
            "NEXA_ABLATE_EMPTY_RESULT_LOOP": params.NEXA_ABLATE_EMPTY_RESULT_LOOP,
            "NEXA_ABLATE_DB_MATCHING": params.NEXA_ABLATE_DB_MATCHING,
            "NEXA_ABLATE_LLM_SELECTION": params.NEXA_ABLATE_LLM_SELECTION,
            "NEXA_ABLATE_SYNTAX_CORRECTION": params.NEXA_ABLATE_SYNTAX_CORRECTION,
            "NEXA_ABLATE_FEWSHOT_EXAMPLES": params.NEXA_ABLATE_FEWSHOT_EXAMPLES,
        },
        **extra,
    }
    with open(manifest_path(out_path), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


############################
# Core runner
############################

def results_dir_for(cfg: RunConfig) -> str:
    """Effective output directory: results/<out_subdir>/ when set, else results/."""
    return os.path.join(RESULTS_DIR, cfg.out_subdir) if cfg.out_subdir else RESULTS_DIR


def output_path_for(file_path: str, bench_type: str, cfg: RunConfig) -> str:
    """Output JSONL path: the dataset name joined with every setting that distinguishes a run."""
    base = os.path.splitext(os.path.basename(file_path))[0]
    # ":" (Ollama tags such as qwen3.8:27b-q8_0) is not allowed in Windows file
    # names; the record keeps the exact model name.
    parts = [base, bench_type, MODEL_NAME.replace(":", "-"), cfg.kg_mode]
    # Tag flex outputs so they never collide with direct-mode files.
    if cfg.agent_mode and cfg.agent_mode != "direct":
        parts.append(cfg.agent_mode)
    if cfg.profile_name:
        parts.append(safe_filename(cfg.profile_name))
    if cfg.examples_mode:
        parts.append(cfg.examples_mode)
    # Generation mode (direct only): otherwise ablation arms would share a file
    # and resume would silently skip their questions.
    if cfg.agent_mode == "direct" and cfg.generation_mode:
        parts.append(safe_filename(cfg.generation_mode))
    if cfg.run_tag:
        parts.append(safe_filename(cfg.run_tag))
    return os.path.join(results_dir_for(cfg), "__".join(parts) + ".jsonl")


# Flex `query()` errors that are diagnostics, not failures: an answer was still
# produced. Stored as `warning`, so --retry-errors and error_rate ignore them.
_SOFT_ERROR_PREFIXES = ("flex: answer produced WITHOUT any execute_cypher",)


def split_soft_error(error: Optional[str], has_answer: bool) -> Tuple[Optional[str], Optional[str]]:
    """Return (error, warning): a known soft diagnostic becomes a warning when an answer exists."""
    if error and has_answer and error.startswith(_SOFT_ERROR_PREFIXES):
        return None, error
    return error, None


def query_with_retries(agent, question: str, schema, prompt: str, cfg: RunConfig):
    """Call agent.query, retrying on raised exceptions only (transient API or network errors).

    Returns (result, attempts, exception or None).
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, cfg.max_retries + 2):
        try:
            res = agent.query(
                user_query=question,
                kg_mode=cfg.kg_mode,
                output_schema=schema,
                custom_system_prompt=prompt,
            )
            return res, attempt, None
        except Exception as e:  # noqa: BLE001  any transient error (rate limit, network)
            last_exc = e
            if attempt <= cfg.max_retries:
                wait = min(30, 2 ** attempt)
                print(f"    attempt {attempt} raised {type(e).__name__}: {e}; retrying in {wait}s")
                time.sleep(wait)
    return None, cfg.max_retries + 1, last_exc


def run_on_file(agent, bench_type: str, file_path: str, cfg: RunConfig) -> str:
    """Run one JSON benchmark file and append JSONL results. Returns output path."""
    type_cfg = BENCH_TYPES[bench_type]
    schema, prompt = type_cfg["schema"], type_cfg["prompt"]

    ensure_dir(results_dir_for(cfg))
    base = os.path.basename(file_path)
    out_path = output_path_for(file_path, bench_type, cfg)
    already = prepare_output_file(out_path, cfg.overwrite, cfg.retry_errors, cfg.retry_all_errors)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if cfg.limit is not None:
        data = data[: cfg.limit]

    started_at = datetime.now(timezone.utc).isoformat()
    write_manifest(out_path, cfg, base, {
        "started_at": started_at,
        "n_questions_in_dataset": len(data),
        "n_skipped_resume": sum(1 for item in data if item.get("question") in already),
    })

    is_flex = isinstance(agent, NexaFlex)
    hops = hops_from_name(base)
    n_done = n_correct = 0

    for idx, item in enumerate(data, start=1):
        question = item["question"]
        expected = item.get("answer", "")
        if question in already:
            continue

        # Fresh in-memory checkpointer per question: the recursion-error reload
        # path still works, and no state carries over between questions.
        if is_flex and cfg.checkpointer == "memory":
            agent._checkpointer = MemorySaver()

        t0 = time.perf_counter()
        with get_usage_metadata_callback() as usage_cb:
            res, attempts, exc = query_with_retries(agent, question, schema, prompt, cfg)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        usage = aggregate_usage(usage_cb.usage_metadata)

        model_answer: Any = "NO_RESPONSE"
        model_answer_text: Optional[str] = None
        thinking = "N/A"
        error: Optional[str] = None
        warning: Optional[str] = None
        # "exception": infrastructure failure (retryable); "system": the agent's
        # own failure on this question (incorrect, not retried).
        error_source: Optional[str] = None

        if exc is not None:
            error = f"{type(exc).__name__}: {exc}"
            error_source = "exception"
        elif res and isinstance(res, dict) and res.get("data"):
            payload = res["data"]
            # Flex forwards a non-dict answer unchanged: the payload is then the answer.
            is_mapping = isinstance(payload, dict)
            model_answer, model_answer_text, error = extract_model_answer(
                bench_type,
                payload.get("answer", []) if is_mapping else payload,
                question,
            )
            thinking = payload.get("thinking", "N/A") if is_mapping else "N/A"
            if not error and res.get("error"):
                error, warning = split_soft_error(str(res.get("error")), has_answer=True)
        else:
            error = (res or {}).get("error") if isinstance(res, dict) else "Invalid agent response"

        record: Dict[str, Any] = {
            "qid": question_id(question),
            "file": base,
            "type": bench_type,
            "hops": hops,
            "question": question,
            "expected": expected,
            "model_answer": model_answer,
            "thinking": thinking,
            "latency_ms": latency_ms,
            "model": MODEL_NAME,
            "provider": BENCHMARK_PROVIDER,
            "agent_mode": cfg.agent_mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if bench_type == "mcq":
            record["model_answer_text"] = model_answer_text
        # Dataset `meta` (e.g. relation path, gold size) is copied for stratified analysis.
        if isinstance(item.get("meta"), dict):
            record["meta"] = item["meta"]
        if cfg.profile_name:
            record["profile"] = cfg.profile_name
        if cfg.agent_mode == "direct":
            if cfg.examples_mode:
                record["examples_mode"] = cfg.examples_mode
            record["generation_mode"] = cfg.generation_mode
        if cfg.run_tag:
            record["run_tag"] = cfg.run_tag
        if usage:
            record["usage"] = usage
        if attempts > 1:
            record["attempts"] = attempts

        if isinstance(res, dict):
            record["cypher_first_attempt"] = res.get("cypher_first_attempt")
            record["cypher_final"] = res.get("cypher_query")
            record["n_correction_iterations"] = res.get("n_correction_iterations")
            n_iter = res.get("n_correction_iterations")
            record["first_attempt_success"] = (n_iter == 0) if isinstance(n_iter, int) else None
            # Direct one_shot_fallback only: the one-shot query came back empty and
            # the decomposed branch ran. None elsewhere.
            record["fallback_triggered"] = res.get("fallback_triggered")
            # Syntax-repair rounds; n_correction_iterations counts property repair only.
            record["n_neo4j_error_corrections"] = res.get("n_neo4j_error_corrections")
            # Flex-only telemetry and full tool trace.
            for k in (
                "trace",
                "n_tool_calls",
                "n_execute_cypher",
                "n_distinct_result_ids",
                "recursion_hit",
                "cited_result_ids",
                # Completeness guard counters (0 when NEXA_COMPLETENESS_GUARD is off).
                "n_truncated_cypher",
                "n_incomplete_results",
                "n_cited_incomplete_results",
                # NEXA_FLEX_ANSWER_RESCUE: True when the answer was rebuilt after
                # the loop failed to produce one (always False when the flag is off).
                "answer_rescued",
            ):
                if k in res:
                    record[k] = res.get(k)

        record["correct"] = evaluate_correctness(bench_type, expected, model_answer)
        if bench_type == "open_ended":
            record["metrics"] = compute_set_metrics(
                csv_to_set(expected), list_to_set(model_answer)
            )
        elif bench_type == "open_ended_list":
            gold_set = entity_list_to_set(expected)
            pred_set = entity_list_to_set(model_answer)
            record["metrics"] = compute_set_metrics(gold_set, pred_set)
            # Gold coverage: every gold entity predicted, extra entities allowed.
            record["metrics"]["gold_covered"] = 1.0 if gold_covered(gold_set, pred_set) else 0.0
            record["n_gold"] = len(gold_set)
            record["n_predicted"] = len(pred_set)
        if error:
            record["error"] = error
            record["error_source"] = error_source or "system"
        if warning:
            record["warning"] = warning

        append_jsonl(out_path, record)

        n_done += 1
        n_correct += 1 if record["correct"] else 0
        tokens = f" tok={usage['total_tokens']}" if usage else ""
        flag = "+" if record["correct"] else ("E" if error else "-")
        print(
            f"  [{idx:>4}/{len(data)}] {flag} acc={n_correct / n_done:.3f} "
            f"{latency_ms / 1000:.1f}s{tokens}  {question[:70]}"
        )

    write_manifest(out_path, cfg, base, {
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "n_questions_in_dataset": len(data),
        "n_run_this_session": n_done,
        "n_correct_this_session": n_correct,
    })
    return out_path


def run_all_of_type(agent, bench_type: str, cfg: RunConfig) -> List[str]:
    """Run every AlzKB dataset of one type; return the output paths."""
    prefix = BENCH_TYPES[bench_type]["prefix"]
    files = sorted(glob.glob(os.path.join(DATASETS_DIR, f"{prefix}*.json")))
    outputs: List[str] = []
    for fp in files:
        print(f"Running {bench_type} on {os.path.basename(fp)} ...")
        outputs.append(run_on_file(agent, bench_type, fp, cfg))
    return outputs


############################
# Agent construction
############################

def ensure_examples_built(profile_name: str, mode: str) -> bool:
    """Build the profile's few-shot examples and identifying properties if missing (direct mode only)."""
    if mode == "hardcoded":
        if load_identifying_properties_for_profile(profile_name):
            return True
    else:
        if load_dynamic_examples_for_profile(profile_name, mode):
            return True

    profile = get_profile_with_password(profile_name)
    if not profile:
        print(f"[{mode}] Profile '{profile_name}' not found; cannot build examples.")
        return False

    nodes_desc, rels_desc = load_descriptions_from_profile(profile, profile_name)
    identifying = (
        profile.get("identifying_properties")
        or load_identifying_properties_from_description(profile)
    )

    llm = None
    if mode in ("llm", "hardcoded"):
        try:
            llm = get_llm(provider=BENCHMARK_PROVIDER, name=MODEL_NAME)
        except Exception as e:
            if mode == "llm":
                print(f"[{mode}] Could not build LLM for examples: {e}")
                return False
            print(f"[{mode}] Could not build LLM (best-effort, continuing): {e}")

    handler = None
    try:
        handler = Neo4jHandler(
            host=profile.get("host"),
            port=profile.get("port"),
            username=profile.get("username"),
            password=profile.get("password"),
            db_name=profile.get("db_name"),
        )
        payload = build_and_store_examples(
            profile_name=profile_name,
            handler=handler,
            nodes_descriptions=nodes_desc,
            relationships_descriptions=rels_desc,
            identifying_properties_user=identifying,
            llm=llm,
            mode=mode,
        )
        return payload is not None
    finally:
        if handler:
            try:
                handler.close()
            except Exception:
                pass


def resolve_examples_mode(profile: Optional[Dict[str, Any]], examples_mode: Optional[str]) -> Optional[str]:
    """Examples mode: the CLI value, else the profile's, else "hardcoded"."""
    if not profile:
        return examples_mode
    return examples_mode or profile.get("examples_mode", "hardcoded")


def init_agent(cfg: RunConfig) -> NexaAgent:
    """Direct-mode agent (Cypher generation and correction pipeline)."""
    if cfg.profile_name:
        profile = get_profile_with_password(cfg.profile_name)
        if not profile:
            raise RuntimeError(f"Profile '{cfg.profile_name}' not found.")

        mode = resolve_examples_mode(profile, cfg.examples_mode)
        if mode not in ("hardcoded", "template", "llm"):
            raise RuntimeError(f"Invalid examples mode: {mode}")
        if not ensure_examples_built(cfg.profile_name, mode):
            raise RuntimeError(
                f"Could not prepare examples for profile='{cfg.profile_name}', mode='{mode}'."
            )
        cfg.examples_mode = mode

        nodes_desc, rels_desc = load_descriptions_from_profile(profile, cfg.profile_name)
        dynamic_examples = load_dynamic_examples_for_profile(cfg.profile_name, mode)
        identifying_properties = load_identifying_properties_for_profile(cfg.profile_name)
        llm = get_llm(provider=BENCHMARK_PROVIDER, name=MODEL_NAME)

        print(
            f"Direct agent initialized: profile={cfg.profile_name}, examples_mode={mode}, "
            f"model={MODEL_NAME}, neo4j={profile.get('host')}:{profile.get('port')}"
        )
        return NexaAgent(
            llm=llm,
            neo4j_host=profile.get("host"),
            neo4j_port=profile.get("port"),
            neo4j_username=profile.get("username"),
            neo4j_password=profile.get("password"),
            neo4j_db_name=profile.get("db_name"),
            nodes_descriptions=nodes_desc,
            relationships_descriptions=rels_desc,
            dynamic_examples=dynamic_examples,
            identifying_properties=identifying_properties,
            generation_mode=cfg.generation_mode,
        )

    print(f"Direct agent initialized: model={MODEL_NAME}, neo4j={params.NEO4J_HOST}:{params.NEO4J_PORT}")
    return NexaAgent.from_credentials(
        model_provider=BENCHMARK_PROVIDER,
        model_name=MODEL_NAME,
        neo4j_host=params.NEO4J_HOST,
        neo4j_port=params.NEO4J_PORT,
        neo4j_username=params.NEO4J_USERNAME,
        neo4j_password=params.NEO4J_PASSWORD,
        neo4j_db_name=params.NEO4J_DB_NAME,
    )


def init_flex_agent(cfg: RunConfig) -> NexaFlex:
    """Flex-mode agent (ReAct loop with granular tools).

    Flex reads the KG schema live, so it needs no examples, only a Postgres
    pool (result store) and the KG profile. Checkpointer "memory" (default):
    `run_on_file` swaps in a fresh MemorySaver per question; "postgres" persists
    every benchmark thread (for debugging a run afterwards).
    """
    pool = NexaAgent.setup_postgres_pool(params.DB_URI)

    checkpointer = None
    if cfg.checkpointer == "postgres":
        checkpointer = PostgresSaver(pool)
        try:
            checkpointer.setup()
        except Exception as e:
            print(f"[flex] checkpointer.setup note (continuing): {e}")
    try:
        ResultStore(pool).setup()
    except Exception as e:
        print(f"[flex] ResultStore.setup note (continuing): {e}")

    agent = NexaFlex.from_config(
        db_uri=params.DB_URI,
        model_provider=BENCHMARK_PROVIDER,
        model_name=MODEL_NAME,
        connection_pool=pool,
        checkpointer=checkpointer,  # memory mode: swapped per-question in run_on_file
        kg_profile_name=cfg.profile_name,
        output_schema_model=None,
    )
    # Stash the recursion cap + pool on the instance: `query` defaults to the
    # cap, and `main` closes the pool in its finally block.
    agent._benchmark_recursion_limit = cfg.recursion_limit
    agent._benchmark_pool = pool

    print(
        f"Flex agent initialized: profile={cfg.profile_name}, model={MODEL_NAME}, "
        f"provider={BENCHMARK_PROVIDER}, recursion_limit={cfg.recursion_limit}, "
        f"checkpointer={cfg.checkpointer}"
    )
    return agent


############################
# CLI
############################

def main():
    parser = argparse.ArgumentParser(
        description="Run the NEXA benchmarks with the direct or flex agent."
    )
    parser.add_argument("--type", choices=list(BENCH_TYPES.keys()),
                        help="Benchmark type: mcq | tf | open_ended (CSV gold) | "
                             "open_ended_list (JSON-list gold, e.g. BioHopR)")
    parser.add_argument("--input",
                        help="Specific input JSON file. If omitted, runs all files of the given type.")
    parser.add_argument("--run-all", action="store_true",
                        help="Run all benchmark files across all types.")
    parser.add_argument("--profile",
                        help="KG profile name from kg_profiles.json.")
    parser.add_argument("--agent-mode", choices=["direct", "flex", "agentic"], default="direct",
                        help="Agent architecture: 'direct' pipeline (default) or 'flex' ReAct loop. "
                             "'agentic' is accepted as an alias for 'flex'.")
    parser.add_argument("--kg-mode", choices=["Off", "Auto", "Force"], default="Force",
                        help="KG tool mode for direct mode (default: Force).")
    parser.add_argument("--examples-mode", choices=["hardcoded", "template", "llm"],
                        help="Direct mode only: few-shot examples mode (defaults to profile setting).")
    parser.add_argument("--generation-mode", default="one_shot_fallback",
                        choices=["decomposed", "one_shot", "one_shot_fallback"],
                        help="Direct mode only: Cypher generation strategy (default one_shot_fallback).")
    parser.add_argument("--recursion-limit", type=int, default=150,
                        help="Flex only: LangGraph iteration cap (safety ceiling, not a target).")
    parser.add_argument("--checkpointer", choices=["memory", "postgres"], default="memory",
                        help="Flex only: 'memory' (default, fresh per question, no DB bloat) "
                             "or 'postgres' (persists benchmark threads for post-hoc debugging).")
    parser.add_argument("--limit", type=int,
                        help="Max questions per dataset (smoke tests).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Truncate matching output files instead of resuming.")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Resume mode that also re-runs infrastructure failures (API/network "
                             "exceptions). System failures (recursion, no answer) stay as incorrect.")
    parser.add_argument("--retry-all-errors", action="store_true",
                        help="Re-run all errored questions, including the agent's own failures. "
                             "This biases accuracy with nondeterministic LLMs: for debugging only, "
                             "never for reported numbers.")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Retries per question on raised exceptions (transient API errors). Default 2.")
    parser.add_argument("--run-tag",
                        help="Tag appended to output filenames (e.g. run1/run2/run3 for repeated runs).")
    parser.add_argument("--out-subdir",
                        help="Write under benchmark/results/<out-subdir>/, normally the arm "
                             "directory <experiment>/<system>_<model> (e.g. alzkb/nexa-flex_gpt-5-mini).")

    args = parser.parse_args()

    # "agentic" is an alias for "flex".
    if args.agent_mode == "agentic":
        args.agent_mode = "flex"

    cfg = RunConfig(
        agent_mode=args.agent_mode,
        kg_mode=args.kg_mode,
        profile_name=args.profile,
        examples_mode=args.examples_mode,
        generation_mode=args.generation_mode,
        recursion_limit=args.recursion_limit,
        checkpointer=args.checkpointer,
        limit=args.limit,
        overwrite=args.overwrite,
        retry_errors=args.retry_errors,
        retry_all_errors=args.retry_all_errors,
        max_retries=args.max_retries,
        run_tag=args.run_tag,
        out_subdir=args.out_subdir,
    )

    # Flex mode reads the schema live and has no few-shot examples, so
    # examples_mode must not leak into its filenames / records.
    if cfg.agent_mode == "flex":
        cfg.examples_mode = None
    else:
        profile = get_profile_with_password(cfg.profile_name) if cfg.profile_name else None
        cfg.examples_mode = resolve_examples_mode(profile, cfg.examples_mode)

    try:
        if cfg.agent_mode == "flex":
            agent = init_flex_agent(cfg)
        else:
            agent = init_agent(cfg)
    except (ValueError, RuntimeError) as e:
        print(f"Error initializing agent: {e}")
        print("Please ensure NEO4J and LLM provider environment variables are set.")
        return

    outputs: List[str] = []
    try:
        if args.run_all:
            for t in BENCH_TYPES.keys():
                outputs.extend(run_all_of_type(agent, t, cfg))
        elif args.type and args.input:
            outputs.append(run_on_file(agent, args.type, args.input, cfg))
        elif args.type:
            outputs.extend(run_all_of_type(agent, args.type, cfg))
        else:
            parser.print_help()
            print("\nNo valid mode specified. Use --type or --run-all.")
            return
    finally:
        # Close the flex Postgres pool (created in init_flex_agent).
        # Direct mode manages its own pool internally.
        pool = getattr(agent, "_benchmark_pool", None)
        if pool is not None:
            try:
                pool.close()
            except Exception:
                pass

    print("\nOutputs written to:")
    for out in outputs:
        print(f"- {out}")


if __name__ == "__main__":
    main()
