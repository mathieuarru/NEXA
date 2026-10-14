"""NEXA configuration, read from environment variables and an optional .env file.

Covers LLM providers, Neo4j and PostgreSQL connections, agent and pipeline flags,
model context windows, and logging/UI settings.
"""
import os

from dotenv import load_dotenv

load_dotenv()

##################  LLM SETTINGS  ##################
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "")
OLLAMA_DEFAULT_MODEL = os.getenv("OLLAMA_DEFAULT_MODEL", "")
BENCHMARK_PROVIDER = os.getenv("BENCHMARK_PROVIDER", "")
BENCHMARK_MODEL_NAME = os.getenv("BENCHMARK_MODEL_NAME", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_DEFAULT_MODEL = os.getenv("ANTHROPIC_DEFAULT_MODEL", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_DEFAULT_MODEL = os.getenv("GOOGLE_DEFAULT_MODEL", "")
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# When true, the user's original question always replaces the user_query
# argument the LLM passes to retrieve_data_from_kg. When false, it is only
# injected if that argument is missing or empty.
NEXA_FORCE_INJECT_USER_QUERY = os.getenv("NEXA_FORCE_INJECT_USER_QUERY", "false").lower() in ("true", "1", "yes")

# Default agent architecture, overridable per request with `agent_mode`:
#   "direct": decompose -> generate -> correct pipeline (NexaAgent)
#   "flex":   tool-calling agent with a result store (NexaFlex)
# Aliases: "pipeline" -> "direct", "agentic" -> "flex".
_AGENT_MODE_RAW = os.getenv("NEXA_AGENT_MODE", "direct").lower()
if _AGENT_MODE_RAW == "pipeline":
    _AGENT_MODE_RAW = "direct"
elif _AGENT_MODE_RAW == "agentic":
    _AGENT_MODE_RAW = "flex"
NEXA_AGENT_MODE = _AGENT_MODE_RAW if _AGENT_MODE_RAW in ("direct", "flex") else "direct"

# Maximum LangGraph steps in a Flex run (about two per tool round-trip).
try:
    NEXA_AGENT_RECURSION_LIMIT = int(os.getenv("NEXA_AGENT_RECURSION_LIMIT", "200"))
except ValueError:
    NEXA_AGENT_RECURSION_LIMIT = 200

# Value matching used by property correction. "composite": DB-side containment
# and token-overlap recall with a composite reranker. "legacy": all distinct
# values ranked by SequenceMatcher only.
NEXA_PROPERTY_MATCHING = os.getenv("NEXA_PROPERTY_MATCHING", "composite").lower()
if NEXA_PROPERTY_MATCHING not in ("composite", "legacy"):
    NEXA_PROPERTY_MATCHING = "composite"

# Mark a Cypher result as incomplete when it returned exactly its own trailing
# `LIMIT n` and more rows exist. Flex treats results as exhaustive, which only
# holds for complete results. Costs one extra query per saturated result.
# Off by default to keep published results reproducible.
NEXA_COMPLETENESS_GUARD = os.getenv("NEXA_COMPLETENESS_GUARD", "false").lower() in ("true", "1", "yes")

# Add the "no LIMIT in an answer query" rule to the Flex system prompt. Kept
# separate from NEXA_COMPLETENESS_GUARD so each can be measured alone. Off by
# default.
NEXA_NO_LIMIT_PROMPT = os.getenv("NEXA_NO_LIMIT_PROMPT", "false").lower() in ("true", "1", "yes")

# When Flex's answer does not match the declared output schema (e.g. a JSON
# string instead of an object), re-emit the same answer in that schema: a
# tolerant parse first, then a constrained-decoding call. A missing answer is
# left missing. Off by default.
NEXA_FLEX_ANSWER_RESCUE = os.getenv(
    "NEXA_FLEX_ANSWER_RESCUE", "false"
).lower() in ("true", "1", "yes")


# Ablation switches: each disables one component (see benchmark/ablation_direct.py).
# All off by default; every run records them in its .meta.json.
def _ablation_flag(name: str) -> bool:
    return os.getenv(name, "false").lower() in ("true", "1", "yes")

# Direct, correction loop. An empty result ends the question: no property
# extraction, no value matching, no LLM rewrite.
NEXA_ABLATE_EMPTY_RESULT_LOOP = _ablation_flag("NEXA_ABLATE_EMPTY_RESULT_LOOP")
# Direct, correction loop. The loop still runs and the LLM still rewrites the
# query, but without the candidate values retrieved from the database.
NEXA_ABLATE_DB_MATCHING = _ablation_flag("NEXA_ABLATE_DB_MATCHING")
# Direct, correction loop. Candidate values are still retrieved from the
# database, but the top-ranked one is taken as is: no LLM contextual selection
# among the candidates and no LLM-proposed synonym.
NEXA_ABLATE_LLM_SELECTION = _ablation_flag("NEXA_ABLATE_LLM_SELECTION")
# Direct, correction loop. The dedicated repair of Neo4j errors and
# notifications is skipped; a query that fails is handled like an empty result.
NEXA_ABLATE_SYNTAX_CORRECTION = _ablation_flag("NEXA_ABLATE_SYNTAX_CORRECTION")
# Direct, all prompts. No few-shot examples at all (neither the profile's
# schema-grounded ones nor the generic placeholder ones).
NEXA_ABLATE_FEWSHOT_EXAMPLES = _ablation_flag("NEXA_ABLATE_FEWSHOT_EXAMPLES")

# Seconds a fetched KG schema is reused before being fetched again; 0 disables
# the cache.
try:
    NEXA_SCHEMA_CACHE_TTL = int(os.getenv("NEXA_SCHEMA_CACHE_TTL", "300"))
except ValueError:
    NEXA_SCHEMA_CACHE_TTL = 300

# Cap on the tokens one LLM call may generate (applied to Ollama, which
# otherwise never stops a repetition loop). Set to about twice the longest
# legitimate answer (long gene lists). A stopping condition, not a sampling
# parameter: calls that finish below the cap are unaffected.
try:
    NEXA_LLM_MAX_OUTPUT_TOKENS = int(os.getenv("NEXA_LLM_MAX_OUTPUT_TOKENS", "32768"))
except ValueError:
    NEXA_LLM_MAX_OUTPUT_TOKENS = 32768

# Tighter cap for the property-suggestion call, which picks one of ten candidate
# values with a one-sentence rationale.
try:
    NEXA_LLM_MAX_PROPERTY_TOKENS = int(os.getenv("NEXA_LLM_MAX_PROPERTY_TOKENS", "1024"))
except ValueError:
    NEXA_LLM_MAX_PROPERTY_TOKENS = 1024

# Context window per model, matched as a case-insensitive substring of the
# model name. The first matching key wins, so each family lists its most
# specific keys first ("qwen3.8" before "qwen3"). Unknown models use
# _DEFAULT_CONTEXT_WINDOW; NEXA_LLM_CONTEXT_WINDOW overrides the lookup.
#
# Open models with a native window of 128K or more are served at 128K, to fit on
# a single 80 GB GPU. The same value sets Ollama's num_ctx, so the token budget
# always matches the served window.
_OPEN_MODEL_CTX = 131_072

_CONTEXT_WINDOWS = {
    # --- OpenAI (most specific first) ---
    "gpt-4o-mini":      128_000,
    "gpt-4o":           128_000,
    "gpt-4-turbo":      128_000,
    "gpt-4":              8_192,
    "gpt-3.5-turbo":     16_385,
    "o1-mini":          128_000,
    "o3-mini":          200_000,
    "gpt-5":            400_000,
    # --- Anthropic (matches every claude-* model) ---
    "claude":           200_000,
    # --- Google (hosted) ---
    "gemini-1.5-pro":  2_000_000,
    "gemini-1.5-flash": 1_000_000,
    "gemini-2.0":       1_000_000,
    # --- Open weights, served by Ollama. Native maximum in comments. ---
    # Gemma 4: edge variants 128K, workstation variants (12b/26b/31b) 256K.
    "gemma4:e":        _OPEN_MODEL_CTX,   # arch 131_072
    "gemma4":          _OPEN_MODEL_CTX,   # arch 262_144
    "gemma3":          _OPEN_MODEL_CTX,   # arch 131_072
    "gemma":              8_192,          # gemma 1/2
    # Qwen 3.5/3.6/3.8 are 256K native; qwen3 is 40_960; qwen2.5 is 32_768.
    "qwen3.8":         _OPEN_MODEL_CTX,   # arch 262_144
    "qwen3.6":         _OPEN_MODEL_CTX,   # arch 262_144
    "qwen3.5":         _OPEN_MODEL_CTX,   # arch 262_144
    "qwen3":             40_960,          # arch 40_960
    "qwen2.5":           32_768,          # arch 32_768
    "qwq":             _OPEN_MODEL_CTX,
    "qwen":              32_768,
    # Llama: 3.1+ are 128K, llama4 far beyond; llama2/3 are small.
    "llama4":          _OPEN_MODEL_CTX,   # arch >= 1M (Scout: 10M)
    "llama3.3":        _OPEN_MODEL_CTX,   # arch 131_072
    "llama3.2":        _OPEN_MODEL_CTX,   # arch 131_072
    "llama3.1":        _OPEN_MODEL_CTX,   # arch 131_072
    "llama3":             8_192,
    "llama2":             4_096,
    "llama":              8_192,
    "deepseek-v4":     _OPEN_MODEL_CTX,   # arch ~1M
    "deepseek":        _OPEN_MODEL_CTX,   # arch 131_072
    "glm":             _OPEN_MODEL_CTX,   # arch >= 131_072
    "devstral":        _OPEN_MODEL_CTX,   # arch 131_072
    "mistral-nemo":    _OPEN_MODEL_CTX,   # arch 131_072
    "mistral-small":   _OPEN_MODEL_CTX,   # arch 131_072
    "mistral-medium":  _OPEN_MODEL_CTX,   # arch 131_072
    "mistral-large":   _OPEN_MODEL_CTX,   # arch 131_072
    "mistral":           32_768,          # 7B v0.3
    "gpt-oss":         _OPEN_MODEL_CTX,   # arch 131_072
}
_DEFAULT_CONTEXT_WINDOW = 8_000

##################  NEO4J SETTINGS  ##################
NEO4J_HOST = os.getenv("NEO4J_HOST", "localhost")
NEO4J_PORT = os.getenv("NEO4J_PORT", "7688")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DB_NAME = os.getenv("NEO4J_DB_NAME", "neo4j")

##################  POSTGRES SETTINGS  ##################
DB_URI = os.getenv("DB_URI")

##################  LOGGING & UI  ##################
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
KG_PROFILES_DIR = os.getenv("KG_PROFILES_DIR", "api_data")
COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"
]
