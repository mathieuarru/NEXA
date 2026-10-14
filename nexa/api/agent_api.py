"""FastAPI backend for NEXA.

Endpoints to list provider models, manage and stream chat threads, manage KG
profiles and agent presets, and inspect Flex traces and stored results.
Conversation state is checkpointed in PostgreSQL.
"""

import uuid
from typing import List, Dict, Optional, Any, Type
from fastapi import FastAPI, Request, HTTPException, Query, Path as FastApiPath, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import json

from neo4j import exceptions
from nexa.core.nexa_agent import NexaAgent
from nexa.flex import NexaFlex
from nexa.flex.trace_reconstruction import reconstruct_trace_from_checkpoint
from nexa.knowledge_graph.cypher_rewrite import rewrite_for_graph
from nexa.result_store import ResultStore
from nexa.api.stop_registry import (
    request_stop,
    is_stop_requested,
    consume_stop,
    clear_stop,
    UserStoppedException,
)
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
import logging
from nexa.knowledge_graph.neo4j_handler import Neo4jHandler
from nexa.knowledge_graph.read_only import policy_notification
from psycopg import errors as pg_errors
from pydantic import BaseModel as PydBaseModel, create_model as pyd_create_model, Field as PydField
from typing import Any as TypingAny, List as TypingList, Dict as TypingDict, Optional as TypingOptional

from nexa.api.profile_manager import (
    KGProfileResponse,
    AllKGProfilesResponse,
    load_profiles_from_disk,
    save_profiles_to_disk,
    save_description_file,
    delete_description_file,
    get_all_profiles_raw,
    get_profile_with_password,
    get_available_examples_modes,
    load_descriptions_from_profile,
    load_identifying_properties_from_description,
    AgentPresetResponse,
    AllAgentPresetsResponse,
    load_agent_presets_from_disk,
    save_agent_presets_to_disk,
    get_all_agent_presets_raw,
    get_agent_preset_raw
)
from nexa.models.dynamic_examples import build_and_store_examples, delete_examples_from_disk

from nexa.config import params
from nexa.utils.message_utils import normalize_message_content
from nexa.core.nexa_agent import get_llm as _build_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


############################
# Config
############################

DB_URI = params.DB_URI
OPENAI_API_KEY = params.OPENAI_API_KEY
ANTHROPIC_API_KEY = params.ANTHROPIC_API_KEY
GOOGLE_API_KEY = params.GOOGLE_API_KEY
AZURE_API_KEY = params.AZURE_API_KEY
AZURE_OPENAI_ENDPOINT = params.AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_VERSION = params.AZURE_OPENAI_API_VERSION
AZURE_OPENAI_CHAT_DEPLOYMENT = params.AZURE_OPENAI_CHAT_DEPLOYMENT

if not DB_URI:
    logger.error("DB_URI environment variable is not defined.")
    raise ValueError("DB_URI must be set in environment variables.")


############################
# FastAPI app
############################

app = FastAPI(
    title="Nexa API",
    description="API for interacting with the Nexa platform and managing chat threads.",
    version="0.1.0"
)

try:
    db_pool = NexaAgent.setup_postgres_pool(DB_URI)
    checkpointer = PostgresSaver(db_pool)

    try:
        checkpointer.setup()
        logger.info("Checkpointer setup completed.")
    except pg_errors.DuplicateColumn:
        logger.warning("Checkpointer setup skipped: column already exists.")
    except Exception as e:
        logger.error(f"Unexpected error during checkpointer setup: {e}", exc_info=True)

    # Flex tables; Direct mode works without them.
    try:
        ResultStore(db_pool).setup()
    except Exception as e:
        logger.error(f"Flex ResultStore setup failed: {e}", exc_info=True)

    logger.info("Database pool and checkpointer initialized successfully.")

except Exception as e:
    logger.critical(f"CRITICAL: Failed to initialize database components: {e}", exc_info=True)


############################
# DB utilities
############################

def list_threads_in_db() -> List[str]:
    """Return all thread IDs from the checkpoint store, ordered by most recent update."""
    try:
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT thread_id FROM (
                        SELECT thread_id, MAX((checkpoint->>'ts')::timestamptz) AS last_update
                        FROM checkpoints WHERE thread_id IS NOT NULL GROUP BY thread_id
                    ) AS t ORDER BY t.last_update DESC
                """)
                rows = cur.fetchall()
        return [row[0] for row in rows]
    except Exception as e:
        logger.error(f"Error while retrieving threads: {e}", exc_info=True)
        return []


def get_checkpoint_data(thread_id: str) -> Optional[Dict]:
    """Return the latest LangGraph checkpoint for a thread, or None if unavailable."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        checkpoint = checkpointer.get(config=config)
        return checkpoint
    except Exception as e:
         logger.error(f"Error retrieving checkpoint for {thread_id}: {e}", exc_info=True)
         return None


############################
# Per-thread agent_mode persistence
############################

def normalize_agent_mode(value: Optional[str]) -> Optional[str]:
    """Map the agent_mode aliases "pipeline" and "agentic" to "direct" and "flex"; other values pass through."""
    if value == "pipeline":
        return "direct"
    if value == "agentic":
        return "flex"
    return value


def record_thread_mode(thread_id: str, agent_mode: str) -> None:
    """Store a thread's agent_mode; no-op if the thread already has one.

    A thread cannot change mode because Direct and Flex store different message structures.
    """
    agent_mode = normalize_agent_mode(agent_mode)
    if agent_mode not in ("direct", "flex"):
        return
    try:
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO nexa_threads (thread_id, agent_mode)
                    VALUES (%s, %s)
                    ON CONFLICT (thread_id) DO NOTHING
                    """,
                    (thread_id, agent_mode),
                )
    except pg_errors.UndefinedTable:
        logger.warning("nexa_threads table missing; skipping thread-mode record.")
    except Exception as e:
        logger.error(f"Failed to record thread mode for {thread_id}: {e}", exc_info=True)


def get_thread_mode(thread_id: str) -> Optional[str]:
    """Return the stored agent_mode of a thread, or None if unknown (the frontend then assumes "direct")."""
    try:
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT agent_mode FROM nexa_threads WHERE thread_id = %s",
                    (thread_id,),
                )
                row = cur.fetchone()
                return normalize_agent_mode(row[0]) if row else None
    except pg_errors.UndefinedTable:
        return None
    except Exception as e:
        logger.error(f"Failed to read thread mode for {thread_id}: {e}", exc_info=True)
        return None


def get_all_thread_modes() -> Dict[str, str]:
    """Return {thread_id: agent_mode} for every known thread."""
    try:
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT thread_id, agent_mode FROM nexa_threads")
                return {row[0]: normalize_agent_mode(row[1]) for row in cur.fetchall()}
    except pg_errors.UndefinedTable:
        return {}
    except Exception as e:
        logger.error(f"Failed to bulk-read thread modes: {e}", exc_info=True)
        return {}


def parse_messages_and_title(checkpoint_data: Optional[Dict]) -> Dict:
    """Extract the message list, title, and KG profile name from a checkpoint's channel values.

    Normalizes each stored message (Human/System/AI/Tool or raw dict) into a
    ``{"role", "content", ...}`` entry. Returns empty defaults when the checkpoint is missing.
    """
    if not checkpoint_data:
        return {"messages": [], "title": None, "kg_profile_name": None}

    channel_values = checkpoint_data.get("channel_values", {})
    messages = channel_values.get("messages", [])
    title = channel_values.get("title", None)
    kg_profile_name = channel_values.get("kg_profile_name", None)
    parsed_messages = []
    for msg in messages:
        role_type = "assistant"
        content = ""
        tool_calls = None
        tool_name: Optional[str] = None
        if isinstance(msg, HumanMessage):
            role_type = "user"
            content = normalize_message_content(msg.content)
        elif isinstance(msg, SystemMessage):
            role_type = "system"
            content = normalize_message_content(msg.content)
        elif isinstance(msg, AIMessage):
            role_type = "assistant"
            content = normalize_message_content(msg.content)
            tool_calls = getattr(msg, 'tool_calls', None) or None
        elif isinstance(msg, ToolMessage):
            role_type = "tool"
            content = msg.content
            tool_name = getattr(msg, 'name', None) or None
        elif isinstance(msg, dict):
            role_type = msg.get("role") or msg.get("type") or "unknown"
            content = msg.get("content", "")
            role_type = role_type.lower()
            if role_type in {"assistant", "user", "system"}:
                content = normalize_message_content(content)
            elif role_type == "tool" and not isinstance(content, str):
                try:
                    content = json.dumps(content, ensure_ascii=False)
                except Exception:
                    content = str(content)
        else:
            role_type = "unknown"
            content = str(msg)

        if role_type == "tool" and not isinstance(content, str):
            try:
                content = json.dumps(content, ensure_ascii=False)
            except Exception:
                content = str(content)
        elif role_type != "tool":
            content = normalize_message_content(content)

        entry = {"role": role_type, "content": content}
        if tool_calls:
            entry["tool_calls"] = tool_calls
        if role_type == "tool" and tool_name:
            entry["name"] = tool_name
        parsed_messages.append(entry)

    return {"messages": parsed_messages, "title": title, "kg_profile_name": kg_profile_name}

############################
# Structured Output helpers
############################

def _map_json_type_to_python(t: str):
    t = (t or '').lower()
    if t in ['string', 'str']:
        return str
    if t in ['integer', 'int']:
        return int
    if t in ['number', 'float', 'double']:
        return float
    if t in ['boolean', 'bool']:
        return bool
    if t in ['array', 'list']:
        return TypingList[TypingAny]
    if t in ['object', 'dict']:
        return TypingDict[str, TypingAny]
    return TypingAny


def build_pydantic_model_from_schema(schema_dict: Dict[str, TypingAny], model_name: str = "UserStructuredOutput") -> Type[PydBaseModel]:
    """Build a Pydantic model from a user-supplied schema for structured output.

    Accepts either a basic JSON Schema object (``type``/``properties``/``required``) or a
    simple ``field: type`` mapping, and returns a dynamically created model class.

    Raises:
        ValueError: If the schema is not a dict or yields no usable fields.
    """

    if not isinstance(schema_dict, dict):
        raise ValueError("structured_output_schema must be a JSON object.")

    fields: Dict[str, tuple] = {}

    if schema_dict.get('type') == 'object' and isinstance(schema_dict.get('properties'), dict):
        properties = schema_dict['properties']
        required_fields = schema_dict.get('required', []) or []
        for field_name, prop_schema in properties.items():
            py_type = TypingAny
            if isinstance(prop_schema, dict):
                t = prop_schema.get('type')
                if t == 'array':
                    items = prop_schema.get('items') or {}
                    item_type = _map_json_type_to_python(items.get('type')) if isinstance(items, dict) else TypingAny
                    py_type = TypingList[item_type]
                else:
                    py_type = _map_json_type_to_python(t)
            elif isinstance(prop_schema, str):
                py_type = _map_json_type_to_python(prop_schema)

            if field_name in required_fields:
                fields[field_name] = (py_type, PydField(...))
            else:
                fields[field_name] = (TypingOptional[py_type], None)
    else:
        for field_name, type_desc in schema_dict.items():
            if isinstance(type_desc, dict) and 'type' in type_desc:
                py_type = _map_json_type_to_python(type_desc.get('type'))
            elif isinstance(type_desc, str):
                py_type = _map_json_type_to_python(type_desc)
            else:
                py_type = TypingAny
            fields[field_name] = (TypingOptional[py_type], None)

    if not fields:
        raise ValueError("No valid fields found in structured_output_schema.")

    return pyd_create_model(model_name, __base__=PydBaseModel, **fields)

############################
# Endpoints
############################

@app.get("/")
def root():
    """Return a welcome message (GET /)."""
    return {"message": "Welcome to the NEXA API!"}


@app.get("/config/defaults", tags=["Config"])
def get_default_models():
    """Return the default model name of each provider, from environment variables."""
    return {
        "openai": params.DEFAULT_MODEL_NAME,
        "anthropic": params.ANTHROPIC_DEFAULT_MODEL,
        "gemini": params.GOOGLE_DEFAULT_MODEL,
        "ollama": params.OLLAMA_DEFAULT_MODEL,
        "azure": params.AZURE_OPENAI_CHAT_DEPLOYMENT,
    }


@app.get("/models/ollama", response_model=Dict[str, List[str]], tags=["Models"])
def get_ollama_models():
    """List the models available on the Ollama server."""
    ollama_api_url = f"{params.OLLAMA_BASE_URL}/api/tags"
    logger.info(f"Requesting Ollama models from {ollama_api_url}")
    try:
        response = requests.get(ollama_api_url, timeout=10)
        response.raise_for_status()

    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error when trying to reach Ollama at {ollama_api_url}")
        raise HTTPException(status_code=503, detail="Could not connect to the Ollama service. Is it running?")
    except requests.exceptions.Timeout:
        logger.error(f"Timeout when trying to reach Ollama at {ollama_api_url}")
        raise HTTPException(status_code=504, detail="Request to Ollama service timed out.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error requesting Ollama models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while communicating with Ollama: {str(e)}")

    try:
        data = response.json()
        models_data = data.get("models", [])
        if not isinstance(models_data, list):
             logger.error(f"Unexpected format from Ollama API. 'models' is not a list: {models_data}")
             raise HTTPException(status_code=500, detail="Received an unexpected format from Ollama API.")

        model_names = [model.get("name") for model in models_data if isinstance(model, dict) and "name" in model]
        model_names = [name for name in model_names if name]

        logger.info(f"Found {len(model_names)} Ollama models.")
        return {"models": sorted(model_names)}

    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON response from Ollama: {response.text}")
        raise HTTPException(status_code=500, detail="Received an invalid JSON response from Ollama.")
    except Exception as e:
        logger.error(f"Unexpected error processing Ollama response: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error processing Ollama response: {str(e)}")


@app.get("/models/openai", response_model=Dict[str, List[str]], tags=["Models"])
def get_openai_models(request: Request):
    """List OpenAI GPT and o-series models.

    The API key comes from the X-OpenAI-Key header, else the environment.
    """
    api_key = request.headers.get("X-OpenAI-Key") or OPENAI_API_KEY

    if not api_key:
        logger.error("OpenAI API key not found (neither in header nor environment variables).")
        raise HTTPException(status_code=400, detail="OpenAI API key is required. Please configure it in Settings > API Keys.")

    openai_api_url = "https://api.openai.com/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    logger.info(f"Requesting OpenAI models from {openai_api_url}")
    try:
        response = requests.get(openai_api_url, headers=headers, timeout=15)
        response.raise_for_status()

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        if status in (401, 403):
            logger.warning(f"OpenAI rejected the API key (status {status}).")
            raise HTTPException(status_code=400, detail="OpenAI API key is invalid or unauthorized. Please check it in Settings > API Keys.")
        logger.error(f"Error requesting OpenAI models: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"An error occurred while communicating with OpenAI API: {str(e)}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error requesting OpenAI models: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"An error occurred while communicating with OpenAI API: {str(e)}")

    try:
        data = response.json()
        models_data = data.get("data", [])
        if not isinstance(models_data, list):
            logger.error(f"Unexpected format from OpenAI API. 'data' is not a list: {models_data}")
            raise HTTPException(status_code=500, detail="Received an unexpected format from OpenAI API.")

        model_names = [model.get("id") for model in models_data if isinstance(model, dict) and "id" in model]
        model_names = [ name for name in model_names if name and (name.startswith('o') or 'gpt' in name) ]


        logger.info(f"Found {len(model_names)} relevant OpenAI models.")
        return {"models": sorted(model_names)}

    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON response from OpenAI: {response.text}")
        raise HTTPException(status_code=500, detail="Received an invalid JSON response from OpenAI.")
    except Exception as e:
        logger.error(f"Unexpected error processing OpenAI response: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error processing OpenAI response: {str(e)}")


@app.get("/models/anthropic", response_model=Dict[str, List[str]], tags=["Models"])
def get_anthropic_models(request: Request):
    """List Anthropic models.

    The API key comes from the X-Anthropic-Key header, else the environment.
    """
    api_key = request.headers.get("X-Anthropic-Key") or ANTHROPIC_API_KEY

    if not api_key:
        logger.error("Anthropic API key not found (neither in header nor environment variables).")
        raise HTTPException(status_code=400, detail="Anthropic API key is required. Please configure it in Settings > API Keys.")

    anthropic_api_url = "https://api.anthropic.com/v1/models"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    logger.info(f"Requesting Anthropic models from {anthropic_api_url}")
    try:
        response = requests.get(anthropic_api_url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        if status in (401, 403):
            logger.warning(f"Anthropic rejected the API key (status {status}).")
            raise HTTPException(status_code=400, detail="Anthropic API key is invalid or unauthorized. Please check it in Settings > API Keys.")
        logger.error(f"Error requesting Anthropic models: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"An error occurred while communicating with Anthropic API: {str(e)}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error requesting Anthropic models: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"An error occurred while communicating with Anthropic API: {str(e)}")

    try:
        data = response.json()
        models_data = data.get("data", [])
        if not isinstance(models_data, list):
            logger.error(f"Unexpected format from Anthropic API. 'data' is not a list: {models_data}")
            raise HTTPException(status_code=500, detail="Received an unexpected format from Anthropic API.")

        model_names = [model.get("id") for model in models_data if isinstance(model, dict) and model.get("id")]
        logger.info(f"Found {len(model_names)} Anthropic models.")
        return {"models": sorted(set(model_names))}

    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON response from Anthropic: {response.text}")
        raise HTTPException(status_code=500, detail="Received an invalid JSON response from Anthropic.")
    except Exception as e:
        logger.error(f"Unexpected error processing Anthropic response: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error processing Anthropic response: {str(e)}")


@app.get("/models/gemini", response_model=Dict[str, List[str]], tags=["Models"])
def get_gemini_models(request: Request):
    """List Gemini models.

    The API key comes from the X-Google-Key header, else the environment.
    """
    api_key = request.headers.get("X-Google-Key") or GOOGLE_API_KEY

    if not api_key:
        logger.error("Google API key not found (neither in header nor environment variables).")
        raise HTTPException(status_code=400, detail="Google API key is required. Please configure it in Settings > API Keys.")

    gemini_api_url = "https://generativelanguage.googleapis.com/v1beta/models"
    params_req = {"key": api_key}
    logger.info(f"Requesting Gemini models from {gemini_api_url}")
    try:
        response = requests.get(gemini_api_url, params=params_req, timeout=15)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        if status in (400, 401, 403):
            logger.warning(f"Gemini rejected the API key (status {status}).")
            raise HTTPException(status_code=400, detail="Google API key is invalid or unauthorized. Please check it in Settings > API Keys.")
        logger.error(f"Error requesting Gemini models: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"An error occurred while communicating with Gemini API: {str(e)}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error requesting Gemini models: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"An error occurred while communicating with Gemini API: {str(e)}")

    try:
        data = response.json()
        models_data = data.get("models", [])
        if not isinstance(models_data, list):
            logger.error(f"Unexpected format from Gemini API. 'models' is not a list: {models_data}")
            raise HTTPException(status_code=500, detail="Received an unexpected format from Gemini API.")

        model_names = []
        for model in models_data:
            if not isinstance(model, dict):
                continue
            model_name = model.get("name")
            if not model_name:
                continue
            short_name = model_name.split("/")[-1]
            model_names.append(short_name)

        logger.info(f"Found {len(model_names)} Gemini models.")
        return {"models": sorted(set(model_names))}

    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON response from Gemini: {response.text}")
        raise HTTPException(status_code=500, detail="Received an invalid JSON response from Gemini.")
    except Exception as e:
        logger.error(f"Unexpected error processing Gemini response: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error processing Gemini response: {str(e)}")


@app.get("/models/azure", response_model=Dict[str, List[str]], tags=["Models"])
def get_azure_models(request: Request):
    """List Azure OpenAI chat deployments.

    Returns AZURE_OPENAI_CHAT_DEPLOYMENT (comma-separated) when set; otherwise
    queries the endpoint, with key and endpoint from headers, else the environment.
    """
    env_models = [m.strip() for m in (AZURE_OPENAI_CHAT_DEPLOYMENT.split(",") if AZURE_OPENAI_CHAT_DEPLOYMENT else []) if m.strip()]
    if env_models:
        logger.info(f"Returning Azure OpenAI models from AZURE_OPENAI_CHAT_DEPLOYMENT: {env_models}")
        return {"models": sorted(set(env_models))}

    api_key = request.headers.get("X-Azure-Key") or AZURE_API_KEY
    endpoint = request.headers.get("X-Azure-Endpoint") or AZURE_OPENAI_ENDPOINT

    if not api_key or not endpoint:
        logger.error("Azure configuration missing (neither in headers nor environment variables).")
        raise HTTPException(status_code=400, detail="Azure API key and endpoint are required. Please configure them in Settings > API Keys.")

    models_url = f"{endpoint.rstrip('/')}/openai/models"
    headers = {"api-key": api_key}
    params_req = {"api-version": AZURE_OPENAI_API_VERSION}
    logger.info(f"Requesting Azure OpenAI models from {models_url}")
    try:
        response = requests.get(models_url, headers=headers, params=params_req, timeout=15)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        if status in (401, 403):
            logger.warning(f"Azure OpenAI rejected the API key (status {status}).")
            raise HTTPException(status_code=400, detail="Azure API key or endpoint is invalid or unauthorized. Please check them in Settings > API Keys.")
        logger.error(f"Error requesting Azure OpenAI models: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"An error occurred while communicating with Azure OpenAI API: {str(e)}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error requesting Azure OpenAI models: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"An error occurred while communicating with Azure OpenAI API: {str(e)}")

    try:
        data = response.json()
        models = data.get("data") or []
        if not isinstance(models, list):
            logger.error(f"Unexpected format from Azure OpenAI API. No list of models: {models}")
            raise HTTPException(status_code=500, detail="Received an unexpected format from Azure OpenAI API.")

        model_ids = []
        for model in models:
            if not isinstance(model, dict):
                continue
            model_id = model.get("id")
            if model_id:
                model_ids.append(str(model_id))

        logger.info(f"Found {len(model_ids)} Azure OpenAI models.")
        return {"models": sorted(set(model_ids))}

    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON response from Azure OpenAI: {response.text}")
        raise HTTPException(status_code=500, detail="Received an invalid JSON response from Azure OpenAI.")
    except Exception as e:
        logger.error(f"Unexpected error processing Azure OpenAI response: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error processing Azure OpenAI response: {str(e)}")


@app.get("/threads", tags=["Threads"])
def list_threads_endpoint():
    """List all conversation threads with their title and agent mode (GET /threads)."""
    threads = list_threads_in_db()
    modes = get_all_thread_modes()
    results = []
    for thread_id in threads:
        checkpoint_data = get_checkpoint_data(thread_id)
        info = parse_messages_and_title(checkpoint_data)
        results.append({
            "thread_id": thread_id,
            "title": info["title"] or f"Conversation {thread_id[:8]}",
            "agent_mode": modes.get(thread_id),
        })
    return {"threads": results}


@app.delete("/threads", tags=["Threads"])
def delete_all_threads_endpoint():
    """Delete every conversation by truncating all checkpoint and thread-metadata tables (DELETE /threads)."""
    tables_to_truncate = [
        "checkpoints",
        "checkpoint_writes",
        "checkpoint_blobs",
        "checkpoint_migrations",
        "nexa_v2_results",  # Flex result store
        "nexa_threads",     # per-thread agent_mode
    ]
    try:
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                for tbl in tables_to_truncate:
                    logger.info(f"Truncating table {tbl}...")
                    try:
                        cur.execute(f"TRUNCATE TABLE {tbl} CASCADE;")
                    except pg_errors.UndefinedTable:
                        logger.info(f"Table {tbl} does not exist; skipping.")
            conn.commit()
        logger.info("All conversation tables truncated.")
        return {"message": "All conversations have been removed."}
    except Exception as e:
        logger.error(f"Error truncating tables: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.post("/threads/create", tags=["Threads"])
def create_new_thread_endpoint(
    request: Request,
    user_message: str,
    model_provider: str = Query(default="openai", description="LLM provider (e.g., 'ollama', 'openai', 'anthropic', 'gemini', 'azure')"),
    model_name: str = Query(default="gpt-4o", description="Specific LLM model name"),
    kg_profile_name: Optional[str] = Query(None, description="Name of the KG Profile to use for Neo4j connection.")
):
    """Create a new thread ID and generate a conversation title from the first user message (POST /threads/create)."""
    if not user_message:
        raise HTTPException(status_code=400, detail="User message cannot be empty.")
    try:
        api_keys = {
            "openai_api_key": request.headers.get("X-OpenAI-Key"),
            "anthropic_api_key": request.headers.get("X-Anthropic-Key"),
            "google_api_key": request.headers.get("X-Google-Key"),
            "azure_api_key": request.headers.get("X-Azure-Key"),
            "azure_endpoint": request.headers.get("X-Azure-Endpoint"),
        }

        agent_instance = NexaAgent.from_config(
            db_uri=DB_URI,
            model_provider=model_provider,
            model_name=model_name,
            kg_profile_name=kg_profile_name,
            connection_pool=db_pool,
            checkpointer=checkpointer,
            **api_keys
        )

        title = agent_instance.generate_conversation_title(first_message=user_message)
        new_thread_id = str(uuid.uuid4())
        logger.info(f"Created new thread_id: {new_thread_id} for model {model_provider}/{model_name} with KG profile: {kg_profile_name}")
        return {"thread_id": new_thread_id, "title": title}

    except ValueError as ve:
        logger.error(f"Configuration error creating thread: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error creating new thread: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/threads/{thread_id}/stop", tags=["Threads"])
def stop_thread_stream_endpoint(thread_id: str):
    """Request a stop of the stream running on ``thread_id``.

    Called from the browser, since the Streamlit script is blocked while it
    streams. The stream loop stops at its next check and saves a stop marker.
    """
    request_stop(thread_id)
    logger.info(f"Stop requested for thread {thread_id}")
    return {"thread_id": thread_id, "stopped": True}


@app.get("/threads/{thread_id}", tags=["Threads"])
def get_thread_endpoint(thread_id: str):
    """Return a thread's title, messages, KG profile, and agent mode (GET /threads/{thread_id})."""
    logger.debug(f"Requesting history for thread_id: {thread_id}")
    checkpoint_data = get_checkpoint_data(thread_id)
    if not checkpoint_data:
         logger.warning(f"Thread history requested for non-existent thread: {thread_id}")
         raise HTTPException(status_code=404, detail="Thread not found.")

    info = parse_messages_and_title(checkpoint_data)
    logger.debug(f"Returning history for thread_id: {thread_id}, Title: {info.get('title')}, KG Profile: {info.get('kg_profile_name')}")
    return {
        "thread_id": thread_id,
        "title": info["title"],
        "messages": info["messages"],
        "kg_profile_name": info["kg_profile_name"],
        "agent_mode": get_thread_mode(thread_id),
    }


@app.post("/threads/{thread_id}/message", tags=["Threads"])
def post_user_message_endpoint(
    thread_id: str,
    user_message: str,
    request: Request,
    model_provider: str = Query(default="openai", description="LLM provider (e.g., 'ollama', 'openai', 'anthropic', 'gemini', 'azure')"),
    model_name: str = Query(default="gpt-4o"),
    kg_mode: str = Query(default="Off", description="Knowledge Graph mode (Off, Auto, Force)"),
    kg_profile_name: Optional[str] = Query(None, description="Name of the KG Profile to use for Neo4j connection."),
    custom_system_prompt: Optional[str] = Query(None, description="A custom system prompt to override the default agent prompt."),
    title: Optional[str] = Query(None, description="Title for the conversation, only used for the first message."),
    structured_output_schema: Optional[str] = Query(None, description="JSON string defining the structured output schema. Accepts a basic JSON Schema or a simple field:type mapping."),
    system_prompt_name: Optional[str] = Query(None, description="Name of a saved system prompt preset to use (server-side)."),
    structured_output_name: Optional[str] = Query(None, description="Name of a saved structured output preset to use (server-side)."),
    examples_mode: Optional[str] = Query(None, description="Override the profile's few-shot examples mode for this request (hardcoded, template, or llm)."),
    generation_mode: Optional[str] = Query(None, description="Cypher generation mode: one_shot_fallback (default), decomposed, or one_shot."),
    agent_mode: Optional[str] = Query(None, description="Agent architecture: 'direct' (deterministic decompose-generate-correct pipeline, default) or 'flex' (autonomous StateGraph with granular tools). The aliases 'pipeline' (for 'direct') and 'agentic' (for 'flex') are also accepted.")
):
    """Post a user message to a thread and stream the agent's response (POST /threads/{thread_id}/message).

    Resolves the system prompt and structured-output schema (inline or from a saved preset), selects
    the direct or Flex agent architecture, records the thread's mode on its first message, and returns
    a ``text/plain`` stream of tagged chunks ([MESSAGES], [THINKING], [TOOL_SUMMARY], [FINAL],
    [STOPPED], [ERROR]) consumed by the frontend.
    """
    if not user_message:
        raise HTTPException(status_code=400, detail="User message cannot be empty.")
    if examples_mode is not None and examples_mode not in ("hardcoded", "template", "llm"):
        raise HTTPException(status_code=400, detail=f"Invalid examples_mode: '{examples_mode}'.")
    if generation_mode is not None and generation_mode not in ("decomposed", "one_shot", "one_shot_fallback"):
        raise HTTPException(status_code=400, detail=f"Invalid generation_mode: '{generation_mode}'.")
    if agent_mode is not None:
        agent_mode = normalize_agent_mode(agent_mode)
        if agent_mode not in ("direct", "flex"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent_mode: '{agent_mode}'. Must be 'direct' or 'flex'.",
            )
    try:
        api_keys = {
            "openai_api_key": request.headers.get("X-OpenAI-Key"),
            "anthropic_api_key": request.headers.get("X-Anthropic-Key"),
            "google_api_key": request.headers.get("X-Google-Key"),
            "azure_api_key": request.headers.get("X-Azure-Key"),
            "azure_endpoint": request.headers.get("X-Azure-Endpoint"),
        }

        # Resolved before building the agent: Flex uses the schema as the type of
        # respond_to_user's `answer` argument.
        resolved_custom_system_prompt = custom_system_prompt
        resolved_structured_output_schema = structured_output_schema

        if not resolved_custom_system_prompt and system_prompt_name:
            preset = get_agent_preset_raw(system_prompt_name)
            if not preset:
                raise HTTPException(status_code=404, detail=f"Agent preset '{system_prompt_name}' not found.")
            sp = preset.get("system_prompt")
            if not sp:
                raise HTTPException(status_code=400, detail=f"Agent preset '{system_prompt_name}' has no system_prompt defined.")
            resolved_custom_system_prompt = sp

        if not resolved_structured_output_schema and structured_output_name:
            preset = get_agent_preset_raw(structured_output_name)
            if not preset:
                raise HTTPException(status_code=404, detail=f"Agent preset '{structured_output_name}' not found.")
            so = preset.get("structured_output_schema")
            if not so:
                raise HTTPException(status_code=400, detail=f"Agent preset '{structured_output_name}' has no structured_output_schema defined.")
            resolved_structured_output_schema = so

        output_schema_model: Optional[Type[PydBaseModel]] = None
        if resolved_structured_output_schema:
            try:
                schema_obj = json.loads(resolved_structured_output_schema)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid structured_output_schema JSON: {str(e)}")
            try:
                output_schema_model = build_pydantic_model_from_schema(schema_obj)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid structured_output_schema format: {str(e)}")

        resolved_agent_mode = (agent_mode or params.NEXA_AGENT_MODE).lower()

        # No-op if the thread already has a mode.
        record_thread_mode(thread_id, resolved_agent_mode)

        if resolved_agent_mode == "flex":
            agent_instance = NexaFlex.from_config(
                db_uri=DB_URI,
                model_provider=model_provider,
                model_name=model_name,
                kg_profile_name=kg_profile_name,
                connection_pool=db_pool,
                checkpointer=checkpointer,
                custom_system_prompt=resolved_custom_system_prompt,
                output_schema_model=output_schema_model,
                output_schema_json=resolved_structured_output_schema,
                **api_keys,
            )
            logger.info(
                f"Streaming for thread {thread_id} [FLEX] using "
                f"{agent_instance._llm.__class__.__name__}, Profile: {kg_profile_name}, "
                f"structured_output={'yes' if output_schema_model else 'no'}"
            )
        else:
            agent_instance = NexaAgent.from_config(
                db_uri=DB_URI,
                model_provider=model_provider,
                model_name=model_name,
                kg_profile_name=kg_profile_name,
                connection_pool=db_pool,
                checkpointer=checkpointer,
                examples_mode_override=examples_mode,
                generation_mode=generation_mode or "one_shot_fallback",
                **api_keys
            )
            logger.info(f"Streaming for thread {thread_id} [DIRECT] using {agent_instance._llm.__class__.__name__} with KG mode: {kg_mode}, Profile: {kg_profile_name}, generation_mode: {generation_mode or 'one_shot_fallback'}")

        def stream_generator():
            """Yield the agent's streamed output as tagged text chunks, honoring cooperative stop requests."""
            # Clear a stale flag left by a previous stream on this thread.
            clear_stop(thread_id)

            stream = agent_instance.stream_agent(
                user_query=user_message,
                thread_id=thread_id,
                kg_mode=kg_mode,
                custom_system_prompt=resolved_custom_system_prompt,
                title=title,
                output_schema=output_schema_model,
                output_schema_json=resolved_structured_output_schema,
                output_schema_name=structured_output_name
            )
            user_stop = False
            try:

                in_think_mode = False

                for chunk in stream:
                    if is_stop_requested(thread_id):
                        user_stop = True
                        logger.info(f"Stream for thread {thread_id} aborted by user stop request.")
                        yield "\n[STOPPED]\n"
                        break

                    ctype = chunk.get("type")
                    if ctype == "updates":
                        yield f"[UPDATES]\n{chunk.get('status', '')}\n{chunk.get('thinking', '')}\n"
                    elif ctype == "tool_call":
                        # Flex trace event. The surrounding newlines keep it on its
                        # own line, since [MESSAGES] chunks carry no newlines.
                        payload = {
                            "name": chunk.get("name"),
                            "summary": chunk.get("summary"),
                            "result_id": chunk.get("result_id"),
                            "args": chunk.get("args") or {},
                        }
                        yield f"\n[TOOL_SUMMARY]{json.dumps(payload, ensure_ascii=False, default=str)}\n"
                    elif ctype == "final":
                        # Flex final answer: `answer_text` (free-form) or `answer`
                        # (structured), plus the cited result ids.
                        payload = {
                            "answer_text": chunk.get("answer_text"),
                            "answer": chunk.get("answer"),
                            "cited_result_ids": chunk.get("cited_result_ids") or [],
                            "highlight_rows": chunk.get("highlight_rows"),
                            "graph_spec": chunk.get("graph_spec"),
                        }
                        yield f"\n[FINAL]{json.dumps(payload, ensure_ascii=False, default=str)}\n"
                    elif ctype == "messages":
                        content = chunk['content']
                        if "<think>" in content:
                            in_think_mode = True
                            yield f"[THINKING]{content}"
                        elif "</think>" in content and in_think_mode:
                            in_think_mode = False
                            yield "[THINKING_COMPLETE]"
                        elif in_think_mode:
                            yield f"[THINKING]{content}"

                        else:
                            yield f"[MESSAGES]{content}"

                logger.info(f"Finished streaming response for thread {thread_id}")
            except UserStoppedException:
                # Stop raised inside the Direct pipeline: same handling as above.
                user_stop = True
                logger.info(f"Stream for thread {thread_id} aborted by user stop request (mid-pipeline).")
                yield "\n[STOPPED]\n"
            except Exception as stream_e:
                logger.error(f"Error during stream generation for thread {thread_id}: {stream_e}", exc_info=True)
                yield f"[ERROR] An error occurred during streaming: {str(stream_e)}"
            finally:
                # Save a stop marker so the stop shows on reload, then clear the flag.
                if user_stop:
                    try:
                        if resolved_agent_mode == "flex":
                            agent_instance.persist_stop_marker(thread_id)
                        else:
                            agent_instance.persist_stop_marker(thread_id, kg_mode=kg_mode)
                    except Exception as persist_e:
                        logger.error(
                            f"Failed to persist stop marker for {thread_id}: {persist_e}",
                            exc_info=True,
                        )
                consume_stop(thread_id)

        return StreamingResponse(stream_generator(), media_type="text/plain")
    except ValueError as ve:
        logger.error(f"Configuration error posting message to {thread_id}: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error setting up stream for message to {thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


############################
# Flex result store endpoints
############################

@app.get("/threads/{thread_id}/trace", tags=["Flex Trace"])
def get_thread_trace_endpoint(thread_id: str):
    """Rebuild a thread's Flex trace from its checkpoint.

    Returns every tool call of the thread in order, in the same shape as the
    live [TOOL_SUMMARY] events. Unknown threads get an empty trace.
    """
    try:
        checkpoint = get_checkpoint_data(thread_id)
        result = reconstruct_trace_from_checkpoint(checkpoint)
        return {"thread_id": thread_id, **result}
    except Exception as e:
        logger.error(f"flex trace reconstruction failed for {thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error reconstructing trace: {str(e)}")


@app.get("/threads/{thread_id}/results", tags=["Flex Results"])
def list_thread_results_endpoint(thread_id: str):
    """List the metadata, without row data, of every Flex result stored for a thread."""
    try:
        store = ResultStore(db_pool)
        items = store.list_by_thread(thread_id)
        return {"thread_id": thread_id, "count": len(items), "results": items}
    except Exception as e:
        logger.error(f"flex list_thread_results failed for {thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing results: {str(e)}")


@app.get("/results/{result_id}", tags=["Flex Results"])
def get_result_endpoint(
    result_id: str,
    limit: Optional[int] = Query(None, ge=1, le=10000, description="If set, return only this many rows (paginated)."),
    offset: int = Query(0, ge=0, description="Row offset when limit is set."),
):
    """Return a stored Flex result: the full record, or one page when ``limit`` is set."""
    try:
        store = ResultStore(db_pool)
        if limit is not None:
            page = store.read(result_id, limit=limit, offset=offset)
            if page is None:
                raise HTTPException(status_code=404, detail=f"Result {result_id} not found")
            return page
        full = store.get(result_id)
        if full is None:
            raise HTTPException(status_code=404, detail=f"Result {result_id} not found")
        return full
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"flex get_result failed for {result_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching result: {str(e)}")


@app.get("/results/{result_id}/graph", tags=["Flex Results"])
def get_result_graph_endpoint(result_id: str):
    """Return the nodes and edges behind a stored Flex result.

    Runs the original Cypher if it returns entities, otherwise a rewrite that
    returns them. Returns an empty graph with a note when there is nothing to
    show, 404 for an unknown result, and 500 on a Neo4j error.
    """
    store = ResultStore(db_pool)
    record = store.get(result_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Result {result_id} not found")

    cypher = record.get("cypher_query")
    thread_id = record.get("thread_id")

    # Use the thread's KG profile, else the default Neo4j settings.
    kg_profile_name = None
    try:
        checkpoint = get_checkpoint_data(thread_id) if thread_id else None
        if checkpoint:
            kg_profile_name = (checkpoint.get("channel_values") or {}).get("kg_profile_name")
    except Exception as e:
        logger.warning(f"flex graph endpoint: could not resolve profile for thread {thread_id}: {e}")

    def _payload(rewritten, aliases, nodes, edges, note, warnings):
        return {
            "result_id": result_id,
            "thread_id": thread_id,
            "kg_profile_name": kg_profile_name,
            "original_query": cypher,
            "rewritten_query": rewritten,
            "aliases": aliases,
            "nodes": nodes,
            "edges": edges,
            "note": note,
            "warnings": warnings or [],
        }

    # Derived results (filter/aggregate) have no Cypher to run.
    if not cypher:
        return _payload(
            None, [], [], [],
            note=(
                f"This result has no original Cypher (source={record.get('source_tool')!r}); "
                "the graph view is only available for execute_cypher results."
            ),
            warnings=[],
        )

    host = port = username = password = db_name = None
    if kg_profile_name:
        profile = get_profile_with_password(kg_profile_name)
        if profile:
            host = profile.get("host")
            port = profile.get("port")
            username = profile.get("username")
            password = profile.get("password")
            db_name = profile.get("db_name")

    handler = Neo4jHandler(
        host=host or params.NEO4J_HOST,
        port=str(port) if port else params.NEO4J_PORT,
        username=username or params.NEO4J_USERNAME,
        password=password or params.NEO4J_PASSWORD,
        db_name=db_name or params.NEO4J_DB_NAME,
    )

    def _coerce(elements):
        if isinstance(elements, dict):
            return elements.get("nodes") or [], elements.get("edges") or []
        return [], []

    try:
        # 1. Use the original query if it returns nodes or relationships. This
        #    also covers unlabeled patterns the rewriter cannot handle.
        try:
            elements, warnings = handler.execute_query_as_elements(cypher)
        except Exception as e:
            logger.warning(f"flex graph endpoint: original query failed as elements: {e}")
            elements, warnings = None, []
        violation = policy_notification(warnings)
        if violation:
            return _payload(None, [], [], [],
                            note=f"Read-only policy: {violation['description']}", warnings=warnings)
        nodes, edges = _coerce(elements)
        if nodes:
            return _payload(None, [], nodes, edges, None, warnings)

        # 2. Scalar projection: rewrite the query to return the entities.
        try:
            rewritten, aliases = rewrite_for_graph(cypher)
        except ValueError as ve:
            # Nothing to visualize, e.g. RETURN count(*).
            return _payload(None, [], [], [], note=str(ve), warnings=[])

        try:
            elements2, warnings2 = handler.execute_query_as_elements(rewritten)
        except Exception as e:
            logger.error(f"flex graph endpoint: Neo4j error on rewritten query: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Neo4j error: {str(e)}")
        nodes2, edges2 = _coerce(elements2)
        violation = policy_notification(warnings2)
        note = (f"Read-only policy: {violation['description']}" if violation else
                None if nodes2 else "The query returned no graph entities to visualize.")
        return _payload(rewritten, aliases, nodes2, edges2, note, warnings2)
    finally:
        try:
            handler.close()
        except Exception:
            pass


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

############################
# Dynamic examples helpers
############################

def _default_llm_for_examples() -> Optional[Any]:
    """Build an LLM for generating profile examples from the first configured provider, or return None."""
    try:
        if OPENAI_API_KEY:
            return _build_llm(provider="openai", name=params.DEFAULT_MODEL_NAME)
        if AZURE_API_KEY and AZURE_OPENAI_ENDPOINT and params.AZURE_OPENAI_CHAT_DEPLOYMENT:
            return _build_llm(provider="azure", name=params.AZURE_OPENAI_CHAT_DEPLOYMENT)
        if ANTHROPIC_API_KEY:
            return _build_llm(provider="anthropic", name=params.ANTHROPIC_DEFAULT_MODEL or "claude-sonnet-4-5")
        if getattr(params, "OLLAMA_DEFAULT_MODEL", None):
            return _build_llm(provider="ollama", name=params.OLLAMA_DEFAULT_MODEL)
    except Exception as e:
        logger.warning(f"Could not build default LLM for examples: {e}")
    return None


def _trigger_examples_build(profile_name: str, profile_data: Dict[str, Any]) -> None:
    """Build and store a profile's few-shot examples and identifying properties.

    Never raises: without built examples the hardcoded defaults are used.
    """
    mode = profile_data.get("examples_mode", "hardcoded")

    nodes_desc, rels_desc = load_descriptions_from_profile(profile_data, profile_name)
    identifying = (
        profile_data.get("identifying_properties")
        or load_identifying_properties_from_description(profile_data)
    )

    # Hardcoded mode still resolves identifying properties, and uses the LLM, if
    # any, to choose between several candidates.
    llm = _default_llm_for_examples() if mode in ("llm", "hardcoded") else None
    if mode == "llm" and llm is None:
        logger.warning(
            f"Profile '{profile_name}' requested mode='llm' but no LLM could be built; "
            "falling back to template-only examples."
        )
        mode = "template"

    handler = None
    try:
        handler = Neo4jHandler(
            host=profile_data.get("host"),
            port=profile_data.get("port"),
            username=profile_data.get("username"),
            password=profile_data.get("password"),
            db_name=profile_data.get("db_name"),
        )
        build_and_store_examples(
            profile_name=profile_name,
            handler=handler,
            nodes_descriptions=nodes_desc,
            relationships_descriptions=rels_desc,
            identifying_properties_user=identifying,
            llm=llm,
            mode=mode,
        )
    except Exception as e:
        logger.warning(f"Dynamic example build failed for profile '{profile_name}': {e}", exc_info=True)
    finally:
        if handler:
            try:
                handler.close()
            except Exception:
                pass


############################
# KG Profile Endpoints
############################

@app.post("/profiles", response_model=KGProfileResponse, status_code=201, tags=["KG Profiles"])
async def create_kg_profile(
    name: str = Form(...),
    host: str = Form(...),
    port: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    db_name: str = Form(...),
    browser_port: Optional[str] = Form("7474"),
    examples_mode: str = Form("hardcoded"),
    identifying_properties: Optional[str] = Form(None),
    description_file: Optional[UploadFile] = File(None)
):
    """Create a new KG connection profile, optionally with a description file, and build its examples (POST /profiles)."""
    profiles_dict_raw = get_all_profiles_raw()

    if name in profiles_dict_raw:
        raise HTTPException(status_code=409, detail=f"Profile with name '{name}' already exists.")

    if examples_mode not in ("hardcoded", "template", "llm"):
        raise HTTPException(status_code=400, detail=f"Invalid examples_mode: '{examples_mode}'.")

    identifying_mapping: Optional[Dict[str, str]] = None
    if identifying_properties:
        try:
            parsed = json.loads(identifying_properties)
            if isinstance(parsed, dict):
                identifying_mapping = {str(k): str(v) for k, v in parsed.items() if v}
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="identifying_properties must be a JSON object.")

    saved_desc_file_name = None
    if description_file:
        if description_file.content_type != "application/json":
            if hasattr(description_file, 'file') and description_file.file:
                description_file.file.close()
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JSON file.")
        saved_desc_file_name = save_description_file(name, description_file)

    new_profile_data = {
        "name": name, "host": host, "port": port, "username": username,
        "password": password, "db_name": db_name, "browser_port": browser_port,
        "description_file_name": saved_desc_file_name,
        "examples_mode": examples_mode,
        "identifying_properties": identifying_mapping,
    }
    profiles_dict_raw[name] = new_profile_data
    save_profiles_to_disk(profiles_dict_raw)

    _trigger_examples_build(name, new_profile_data)

    response_data = new_profile_data.copy()
    del response_data["password"]
    return KGProfileResponse(**response_data)


@app.get("/profiles", response_model=AllKGProfilesResponse, tags=["KG Profiles"])
async def get_all_kg_profiles():
    """Return all KG connection profiles, passwords omitted (GET /profiles)."""
    profiles = load_profiles_from_disk()
    return AllKGProfilesResponse(profiles=list(profiles.values()))


@app.get("/profiles/{profile_name}", response_model=KGProfileResponse, tags=["KG Profiles"])
async def get_kg_profile(profile_name: str = FastApiPath(..., title="The name of the KG profile")):
    """Return a single KG connection profile by name (GET /profiles/{profile_name})."""
    profiles = load_profiles_from_disk()
    profile = profiles.get(profile_name)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_name}' not found.")
    return profile


@app.put("/profiles/{profile_name}", response_model=KGProfileResponse, tags=["KG Profiles"])
async def update_kg_profile(
    profile_name: str = FastApiPath(..., title="The name of the KG profile to update"),
    host: Optional[str] = Form(None),
    port: Optional[str] = Form(None),
    username: Optional[str] = Form(None),
    password: Optional[str] = Form(None),
    db_name: Optional[str] = Form(None),
    browser_port: Optional[str] = Form(None),
    examples_mode: Optional[str] = Form(None),
    identifying_properties: Optional[str] = Form(None),
    description_file: Optional[UploadFile] = File(None)
):
    """Update fields of an existing KG profile and rebuild its examples when changes are applied (PUT /profiles/{profile_name})."""
    profiles_dict_raw = get_all_profiles_raw()
    existing_profile_data = profiles_dict_raw.get(profile_name)

    if not existing_profile_data:
        if description_file and hasattr(description_file, 'file') and description_file.file:
            description_file.file.close()
        raise HTTPException(status_code=404, detail=f"Profile '{profile_name}' not found.")

    if examples_mode is not None and examples_mode not in ("hardcoded", "template", "llm"):
        raise HTTPException(status_code=400, detail=f"Invalid examples_mode: '{examples_mode}'.")

    identifying_mapping: Optional[Dict[str, str]] = None
    if identifying_properties is not None:
        try:
            parsed = json.loads(identifying_properties) if identifying_properties else None
            if parsed is None:
                identifying_mapping = None
            elif isinstance(parsed, dict):
                identifying_mapping = {str(k): str(v) for k, v in parsed.items() if v}
            else:
                raise HTTPException(status_code=400, detail="identifying_properties must be a JSON object.")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="identifying_properties must be a JSON object.")

    update_data = {
        "host": host, "port": port, "username": username,
        "password": password, "db_name": db_name, "browser_port": browser_port,
        "examples_mode": examples_mode,
    }

    updated_profile_data = existing_profile_data.copy()
    changed = False
    for key, value in update_data.items():
        if value is not None:
            updated_profile_data[key] = value
            changed = True

    if identifying_properties is not None:
        updated_profile_data["identifying_properties"] = identifying_mapping
        changed = True

    if description_file:
        if description_file.content_type != "application/json":
            if hasattr(description_file, 'file') and description_file.file:
                description_file.file.close()
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JSON file.")
        delete_description_file(existing_profile_data.get("description_file_name"))
        updated_profile_data["description_file_name"] = save_description_file(profile_name, description_file)
        changed = True

    if changed:
        profiles_dict_raw[profile_name] = updated_profile_data
        save_profiles_to_disk(profiles_dict_raw)
        _trigger_examples_build(profile_name, updated_profile_data)

    response_data = updated_profile_data.copy()
    del response_data["password"]
    return KGProfileResponse(**response_data)


@app.post("/profiles/{profile_name}/rebuild_examples", tags=["KG Profiles"])
async def rebuild_kg_profile_examples(
    profile_name: str = FastApiPath(..., title="The name of the KG profile to rebuild examples for"),
):
    """Rebuild a profile's few-shot examples, e.g. after a schema change."""
    profile_data = get_profile_with_password(profile_name)
    if not profile_data:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_name}' not found.")

    mode = profile_data.get("examples_mode", "hardcoded")
    if mode == "hardcoded":
        return {"status": "skipped", "reason": "profile uses examples_mode=hardcoded"}

    _trigger_examples_build(profile_name, profile_data)
    return {"status": "ok", "mode": mode}


@app.post("/schema-cache/refresh", tags=["KG Profiles"])
async def refresh_schema_cache():
    """Clear the cached KG schemas so the next question reads them again.

    Cached schemas are revalidated with a fingerprint (counts, label and type
    lists); this covers changes it misses, such as a new property on existing nodes.
    """
    from nexa.knowledge_graph.graph_query import clear_schema_cache

    clear_schema_cache()
    logger.info("Schema cache cleared on request.")
    return {"status": "ok", "detail": "Schema cache cleared; it will be re-read on the next question."}


@app.get("/profiles/{profile_name}/available_examples_modes", tags=["KG Profiles"])
async def get_profile_available_examples_modes(
    profile_name: str = FastApiPath(..., title="The name of the KG profile"),
):
    """Return the usable few-shot examples modes of a profile and its default mode."""
    profiles_raw = get_all_profiles_raw()
    if profile_name not in profiles_raw:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_name}' not found.")
    return get_available_examples_modes(profile_name)


@app.delete("/profiles/{profile_name}", status_code=204, tags=["KG Profiles"])
async def delete_kg_profile(profile_name: str = FastApiPath(..., title="The name of the KG profile to delete")):
    """Delete a KG profile along with its description file and cached examples (DELETE /profiles/{profile_name})."""
    profiles_dict_raw = get_all_profiles_raw()
    profile_to_delete = profiles_dict_raw.get(profile_name)

    if not profile_to_delete:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_name}' not found.")

    delete_description_file(profile_to_delete.get("description_file_name"))
    delete_examples_from_disk(profile_name)
    del profiles_dict_raw[profile_name]
    save_profiles_to_disk(profiles_dict_raw)
    return None


@app.get("/profiles/{profile_name}/styles", response_model=Dict[str, str], tags=["KG Profiles"])
async def get_kg_profile_label_styles(profile_name: str = FastApiPath(..., title="The name of the KG profile")):
    """Return a color mapping for each node label in the profile's graph (GET /profiles/{profile_name}/styles).

    Connects to the profile's Neo4j instance, collects the distinct node labels, and assigns each a
    color from the configured palette.
    """
    logger.info(f"Requesting label styles for KG profile: {profile_name}")
    profile_data_with_pwd = get_profile_with_password(profile_name)
    if not profile_data_with_pwd:
        raise HTTPException(status_code=404, detail=f"KG Profile '{profile_name}' not found.")

    handler = None
    try:
        handler = Neo4jHandler(
            host=profile_data_with_pwd.get("host"),
            port=profile_data_with_pwd.get("port"),
            username=profile_data_with_pwd.get("username"),
            password=profile_data_with_pwd.get("password"),
            db_name=profile_data_with_pwd.get("db_name")
        )

        nodes_meta = handler.get_nodes()

        if nodes_meta is None:
            logger.error(f"Failed to fetch node metadata for profile {profile_name} using get_nodes(). Result was None.")
            raise HTTPException(status_code=500, detail=f"Failed to fetch node metadata for profile {profile_name}.")

        unique_labels = set()
        for item in nodes_meta:
            if isinstance(item, dict) and "node" in item and isinstance(item["node"], dict) and "labels" in item["node"]:
                labels_data = item["node"]["labels"]
                if isinstance(labels_data, str):
                    unique_labels.add(labels_data)
                elif isinstance(labels_data, list):
                    for label_in_list in labels_data:
                        if isinstance(label_in_list, str):
                            unique_labels.add(label_in_list)

        sorted_node_labels = sorted(list(unique_labels))

        label_color_mapping = {}
        if not sorted_node_labels:
            logger.warning(f"No node labels found for profile {profile_name}. Returning empty style map.")
        else:
            for i, label in enumerate(sorted_node_labels):
                label_color_mapping[label] = params.COLOR_PALETTE[i % len(params.COLOR_PALETTE)]

        logger.info(f"Successfully generated {len(label_color_mapping)} label styles for profile {profile_name} using get_nodes().")
        return label_color_mapping

    except exceptions.ServiceUnavailable as e:
        logger.error(f"Neo4j service unavailable for profile {profile_name}: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Neo4j service unavailable for profile '{profile_name}': {str(e)}")
    except exceptions.AuthError as e:
        logger.error(f"Neo4j authentication failed for profile {profile_name}: {e}", exc_info=True)
        raise HTTPException(status_code=401, detail=f"Neo4j authentication failed for profile '{profile_name}'. Check credentials.")
    except Exception as e:
        logger.error(f"Error generating label styles for profile {profile_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating styles for profile '{profile_name}': {str(e)}")
    finally:
        if handler:
            handler.close()


############################
# Agent Presets Endpoints
############################

@app.post("/agent-presets", response_model=AgentPresetResponse, status_code=201, tags=["Agent Presets"])
async def create_agent_preset(
    name: str = Form(...),
    system_prompt: Optional[str] = Form(None),
    structured_output_schema: Optional[str] = Form(None)
):
    """Create a named agent preset holding a system prompt and/or structured-output schema (POST /agent-presets)."""
    presets_raw = get_all_agent_presets_raw()
    if name in presets_raw:
        raise HTTPException(status_code=409, detail=f"Agent preset with name '{name}' already exists.")

    presets_raw[name] = {
        "name": name,
        "system_prompt": system_prompt,
        "structured_output_schema": structured_output_schema
    }
    save_agent_presets_to_disk(presets_raw)
    return AgentPresetResponse(name=name, system_prompt=system_prompt, structured_output_schema=structured_output_schema)

@app.get("/agent-presets", response_model=AllAgentPresetsResponse, tags=["Agent Presets"])
async def get_all_agent_presets():
    """Return all saved agent presets (GET /agent-presets)."""
    presets = load_agent_presets_from_disk()
    return AllAgentPresetsResponse(presets=list(presets.values()))


@app.get("/agent-presets/{preset_name}", response_model=AgentPresetResponse, tags=["Agent Presets"])
async def get_agent_preset(preset_name: str = FastApiPath(..., title="The name of the agent preset")):
    """Return a single agent preset by name (GET /agent-presets/{preset_name})."""
    presets = load_agent_presets_from_disk()
    preset = presets.get(preset_name)
    if not preset:
        raise HTTPException(status_code=404, detail=f"Agent preset '{preset_name}' not found.")
    return preset


@app.put("/agent-presets/{preset_name}", response_model=AgentPresetResponse, tags=["Agent Presets"])
async def update_agent_preset(
    preset_name: str = FastApiPath(..., title="The name of the agent preset to update"),
    system_prompt: Optional[str] = Form(None),
    structured_output_schema: Optional[str] = Form(None)
):
    """Update the system prompt and/or structured-output schema of an existing agent preset (PUT /agent-presets/{preset_name})."""
    presets_raw = get_all_agent_presets_raw()
    existing = presets_raw.get(preset_name)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Agent preset '{preset_name}' not found.")

    if system_prompt is not None:
        existing["system_prompt"] = system_prompt
    if structured_output_schema is not None:
        existing["structured_output_schema"] = structured_output_schema

    presets_raw[preset_name] = existing
    save_agent_presets_to_disk(presets_raw)
    return AgentPresetResponse(name=preset_name, system_prompt=existing.get("system_prompt"), structured_output_schema=existing.get("structured_output_schema"))


@app.delete("/agent-presets/{preset_name}", status_code=204, tags=["Agent Presets"])
async def delete_agent_preset(preset_name: str = FastApiPath(..., title="The name of the agent preset to delete")):
    """Delete an agent preset by name (DELETE /agent-presets/{preset_name})."""
    presets_raw = get_all_agent_presets_raw()
    if preset_name not in presets_raw:
        raise HTTPException(status_code=404, detail=f"Agent preset '{preset_name}' not found.")
    del presets_raw[preset_name]
    save_agent_presets_to_disk(presets_raw)
    return None
