"""HTTP client for the NEXA API used by the Streamlit frontend."""

import streamlit as st
import requests
import json
import os
from typing import List, Dict, Optional

from dotenv import load_dotenv
load_dotenv()

API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")


def get_api_keys_headers() -> Dict[str, str]:
    """Build request headers carrying the API keys entered in the settings page.

    They override the server's environment keys. Read from the `*_stored`
    session keys, which persist across page navigation.
    """
    headers = {}

    openai_key = st.session_state.get("openai_api_key_stored", "")
    anthropic_key = st.session_state.get("anthropic_api_key_stored", "")
    google_key = st.session_state.get("google_api_key_stored", "")
    azure_key = st.session_state.get("azure_api_key_stored", "")
    azure_endpoint = st.session_state.get("azure_endpoint_stored", "")

    if openai_key:
        headers["X-OpenAI-Key"] = openai_key
    if anthropic_key:
        headers["X-Anthropic-Key"] = anthropic_key
    if google_key:
        headers["X-Google-Key"] = google_key
    if azure_key:
        headers["X-Azure-Key"] = azure_key
    if azure_endpoint:
        headers["X-Azure-Endpoint"] = azure_endpoint

    return headers


def safe_request(method: str, url: str, **kwargs) -> Optional[requests.Response]:
    """Send an HTTP request; return None on failure, showing errors other than 4xx."""
    try:
        resp = requests.request(method, url, **kwargs)
        resp.raise_for_status()
        return resp
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        if status is not None and 400 <= status < 500:
            return None
        st.error(f"API error ({method} {url}): {e}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"API error ({method} {url}): {e}")
        return None


@st.cache_data(ttl=3600)  # Cache for 1 hour (defaults rarely change)
def get_default_models() -> Dict[str, str]:
    """Fetch default models for each provider from the API."""
    resp = safe_request("get", f"{API_BASE_URL}/config/defaults")
    if resp:
        try:
            return resp.json()
        except json.JSONDecodeError:
            pass
    # Fallback defaults if API fails
    return {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-sonnet-latest",
        "gemini": "gemini-1.5-pro-latest",
        "ollama": "",
        "azure": "",
    }


def list_threads() -> List[Dict]:
    """Fetch the list of all existing threads."""
    resp = safe_request("get", f"{API_BASE_URL}/threads")
    if resp:
        try:
            data = resp.json()
            return data.get("threads", [])
        except json.JSONDecodeError:
            st.error("Invalid API response (list_threads).")
            return []
    return []

def get_thread_history(thread_id: str) -> Optional[Dict]:
    """Fetch the full history of a thread."""
    resp = safe_request("get", f"{API_BASE_URL}/threads/{thread_id}")
    if resp:
        try:
            return resp.json()
        except json.JSONDecodeError:
             st.error(f"Invalid API response (get_thread_history for {thread_id}).")
             return None
    return None

def create_new_thread(user_message: str, model_provider: str, model_name: str) -> Optional[Dict]:
    """Create a new thread via /threads/create."""
    params = {
        "user_message": user_message,
        "model_provider": model_provider,
        "model_name": model_name,
        "kg_profile_name": st.session_state.get("selected_kg_profile_name")
    }
    headers = get_api_keys_headers()

    resp = safe_request(
        "post",
        f"{API_BASE_URL}/threads/create",
        params=params,
        headers=headers
    )
    if resp:
        try:
            return resp.json()
        except json.JSONDecodeError:
             st.error("Invalid API response (create_new_thread).")
             return None
    st.error(f"Failed to create thread for message: {user_message[:50]}...")
    return None

def stream_message_to_thread(thread_id: str, user_message: str, model_provider: str, model_name: str, callback, title: Optional[str] = None):
    """Send a message to a thread and yield the displayable text of the streamed reply."""
    endpoint_url = f"{API_BASE_URL}/threads/{thread_id}/message"
    kg_mode = st.session_state.get("kg_mode", "Auto")

    params = {
        "user_message": user_message,
        "model_provider": model_provider,
        "model_name": model_name,
        "kg_mode": kg_mode,
        "kg_profile_name": st.session_state.get("selected_kg_profile_name")
    }
    if title:
        params["title"] = title

    preset_prompt_name = st.session_state.get("selected_system_prompt_name")
    preset_schema_name = st.session_state.get("selected_structured_output_name")
    if preset_prompt_name:
        params["system_prompt_name"] = preset_prompt_name
    if preset_schema_name:
        params["structured_output_name"] = preset_schema_name

    examples_mode_override = st.session_state.get("selected_examples_mode_override")
    if examples_mode_override:
        params["examples_mode"] = examples_mode_override

    # one_shot_fallback is the backend default; forward any other choice.
    generation_mode = st.session_state.get("selected_generation_mode")
    if generation_mode and generation_mode != "one_shot_fallback":
        params["generation_mode"] = generation_mode

    agent_mode = st.session_state.get("selected_agent_mode")
    # "direct" is the default and "pipeline" its alias.
    if agent_mode and agent_mode not in ("direct", "pipeline"):
        params["agent_mode"] = agent_mode

    try:
        headers = {"Accept": "text/plain"}
        headers.update(get_api_keys_headers())

        with requests.post(
            endpoint_url,
            params=params,
            stream=True,
            headers=headers,
        ) as resp:
            try:
                resp.raise_for_status()
            except requests.exceptions.HTTPError as http_err:
                 error_detail = resp.text
                 st.error(f"API error ({resp.status_code}) during streaming: {error_detail or http_err}")
                 return

            for chunk in resp.iter_content(chunk_size=None):
                if chunk:
                    decoded = chunk.decode("utf-8")
                    chunk_processed = callback.process_chunk(decoded)
                    if chunk_processed is not None:
                        yield chunk_processed

            callback.complete()

    except requests.exceptions.RequestException as e:
        st.error(f"Connection error during streaming from {endpoint_url}: {e}")
    except Exception as e:
         st.error(f"Unexpected error during streaming: {e}")

@st.cache_data(ttl=300)
def get_ollama_models_from_api() -> List[str]:
    """Fetch available Ollama models from the backend API."""
    resp = safe_request("get", f"{API_BASE_URL}/models/ollama")
    if resp:
        try:
            data = resp.json()
            models = data.get("models", [])
            if isinstance(models, list):
                return sorted([str(model_name) for model_name in models if isinstance(model_name, str)])
            else:
                st.warning("API /models/ollama did not return a valid list.")
                return []
        except json.JSONDecodeError:
            st.error("Invalid API response (get_ollama_models_from_api).")
            return []
        except Exception as e:
             st.error(f"Error fetching Ollama models: {e}")
             return []
    return []

@st.cache_data(ttl=300)
def get_openai_models_from_api(api_key: str = "") -> List[str]:
    """Fetch available OpenAI models from the backend API."""
    headers = {"X-OpenAI-Key": api_key} if api_key else {}
    resp = safe_request("get", f"{API_BASE_URL}/models/openai", headers=headers)
    if resp:
        try:
            data = resp.json()
            models = data.get("models", [])
            if isinstance(models, list):
                return sorted([str(model_name) for model_name in models if isinstance(model_name, str)])
            else:
                st.warning("API /models/openai did not return a valid list.")
                return []
        except json.JSONDecodeError:
            st.error("Invalid API response (get_openai_models_from_api).")
            return []
        except Exception as e:
            st.error(f"Error fetching OpenAI models: {e}")
            return []
    return []


@st.cache_data(ttl=300)
def get_anthropic_models_from_api(api_key: str = "") -> List[str]:
    """Fetch available Anthropic models from the backend API."""
    headers = {"X-Anthropic-Key": api_key} if api_key else {}
    resp = safe_request("get", f"{API_BASE_URL}/models/anthropic", headers=headers)
    if resp:
        try:
            data = resp.json()
            models = data.get("models", [])
            if isinstance(models, list):
                return sorted([str(model_name) for model_name in models if isinstance(model_name, str)])
            else:
                st.warning("API /models/anthropic did not return a valid list.")
                return []
        except json.JSONDecodeError:
            st.error("Invalid API response (get_anthropic_models_from_api).")
            return []
        except Exception as e:
            st.error(f"Error fetching Anthropic models: {e}")
            return []
    return []


@st.cache_data(ttl=300)
def get_gemini_models_from_api(api_key: str = "") -> List[str]:
    """Fetch available Gemini models from the backend API."""
    headers = {"X-Google-Key": api_key} if api_key else {}
    resp = safe_request("get", f"{API_BASE_URL}/models/gemini", headers=headers)
    if resp:
        try:
            data = resp.json()
            models = data.get("models", [])
            if isinstance(models, list):
                return sorted([str(model_name) for model_name in models if isinstance(model_name, str)])
            else:
                st.warning("API /models/gemini did not return a valid list.")
                return []
        except json.JSONDecodeError:
            st.error("Invalid API response (get_gemini_models_from_api).")
            return []
        except Exception as e:
            st.error(f"Error fetching Gemini models: {e}")
            return []
    return []


@st.cache_data(ttl=300)
def get_azure_models_from_api(api_key: str = "", endpoint: str = "") -> List[str]:
    """Fetch available Azure OpenAI deployments from the backend API."""
    headers = {}
    if api_key:
        headers["X-Azure-Key"] = api_key
    if endpoint:
        headers["X-Azure-Endpoint"] = endpoint
    resp = safe_request("get", f"{API_BASE_URL}/models/azure", headers=headers)
    if resp:
        try:
            data = resp.json()
            models = data.get("models", [])
            if isinstance(models, list):
                return sorted([str(model_name) for model_name in models if isinstance(model_name, str)])
            else:
                st.warning("API /models/azure did not return a valid list.")
                return []
        except json.JSONDecodeError:
            st.error("Invalid API response (get_azure_models_from_api).")
            return []
        except Exception as e:
            st.error(f"Error fetching Azure models: {e}")
            return []
    return []

def delete_all_threads() -> bool:
    """Delete all conversations via the API."""
    resp = safe_request("delete", f"{API_BASE_URL}/threads")
    if resp:
        st.success("All conversations have been deleted.")
        return True
    else:
        return False

def get_kg_profiles() -> List[Dict]:
    """Fetch the list of KG profiles from the API."""
    resp = safe_request("get", f"{API_BASE_URL}/profiles")
    if resp:
        try:
            data = resp.json()
            return data.get("profiles", [])
        except json.JSONDecodeError:
            st.error("Invalid API response (get_kg_profiles).")
            return []
    return []

def list_agent_presets() -> List[Dict]:
    """Fetch the agent presets (system prompts and output schemas)."""
    resp = safe_request("get", f"{API_BASE_URL}/agent-presets")
    if resp:
        try:
            data = resp.json()
            return data.get("presets", [])
        except json.JSONDecodeError:
            st.error("Invalid API response (list_agent_presets).")
            return []
    return []

def create_agent_preset(name: str, system_prompt: Optional[str], structured_output_schema: Optional[str]) -> Optional[Dict]:
    """Create an agent preset via the API."""
    data = {"name": name}
    if system_prompt is not None:
        data["system_prompt"] = system_prompt
    if structured_output_schema is not None:
        data["structured_output_schema"] = structured_output_schema
    resp = safe_request("post", f"{API_BASE_URL}/agent-presets", data=data)
    if resp:
        try:
            return resp.json()
        except json.JSONDecodeError:
            st.error("Invalid API response (create_agent_preset).")
            return None
    return None

def update_agent_preset(name: str, system_prompt: Optional[str] = None, structured_output_schema: Optional[str] = None) -> Optional[Dict]:
    """Update an agent preset via the API; fields left as None are unchanged."""
    data: Dict[str, str] = {}
    if system_prompt is not None:
        data["system_prompt"] = system_prompt
    if structured_output_schema is not None:
        data["structured_output_schema"] = structured_output_schema
    resp = safe_request("put", f"{API_BASE_URL}/agent-presets/{name}", data=data)
    if resp:
        try:
            return resp.json()
        except json.JSONDecodeError:
            st.error("Invalid API response (update_agent_preset).")
            return None
    return None

def delete_agent_preset(name: str) -> bool:
    """Delete an agent preset via the API."""
    resp = safe_request("delete", f"{API_BASE_URL}/agent-presets/{name}")
    return resp is not None

def get_kg_profile(profile_name: str) -> Optional[Dict]:
    """Fetch a specific KG profile from the API."""
    resp = safe_request("get", f"{API_BASE_URL}/profiles/{profile_name}")
    if resp:
        try:
            return resp.json()
        except json.JSONDecodeError:
            st.error(f"Invalid API response (get_kg_profile for {profile_name}).")
            return None
    return None

def create_kg_profile(
    name: str,
    host: str,
    port: str,
    username: str,
    password: str,
    db_name: str,
    browser_port: str = "7474",
    description_file: Optional[bytes] = None,
    examples_mode: str = "hardcoded",
    identifying_properties: Optional[str] = None,
) -> Optional[Dict]:
    """Create a new KG profile via the API."""
    files = {}
    if description_file:
        files["description_file"] = ("description.json", description_file, "application/json")

    data = {
        "name": name,
        "host": host,
        "port": port,
        "username": username,
        "password": password,
        "db_name": db_name,
        "browser_port": browser_port,
        "examples_mode": examples_mode,
    }
    if identifying_properties:
        data["identifying_properties"] = identifying_properties

    resp = safe_request(
        "post",
        f"{API_BASE_URL}/profiles",
        data=data,
        files=files
    )
    if resp:
        try:
            return resp.json()
        except json.JSONDecodeError:
            st.error("Invalid API response (create_kg_profile).")
            return None
    return None

def update_kg_profile(
    profile_name: str,
    host: Optional[str] = None,
    port: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    db_name: Optional[str] = None,
    browser_port: Optional[str] = None,
    description_file: Optional[bytes] = None,
    examples_mode: Optional[str] = None,
    identifying_properties: Optional[str] = None,
) -> Optional[Dict]:
    """Update an existing KG profile via the API."""
    data = {}
    if host is not None: data["host"] = host
    if port is not None: data["port"] = port
    if username is not None: data["username"] = username
    if password is not None: data["password"] = password
    if db_name is not None: data["db_name"] = db_name
    if browser_port is not None: data["browser_port"] = browser_port
    if examples_mode is not None: data["examples_mode"] = examples_mode
    if identifying_properties is not None: data["identifying_properties"] = identifying_properties

    files = {}
    if description_file:
        files["description_file"] = ("description.json", description_file, "application/json")

    resp = safe_request(
        "put",
        f"{API_BASE_URL}/profiles/{profile_name}",
        data=data,
        files=files
    )
    if resp:
        try:
            return resp.json()
        except json.JSONDecodeError:
            st.error("Invalid API response (update_kg_profile).")
            return None
    return None


def rebuild_kg_profile_examples(profile_name: str) -> Optional[Dict]:
    """Force a rebuild of the dynamic few-shot examples for a profile."""
    resp = safe_request("post", f"{API_BASE_URL}/profiles/{profile_name}/rebuild_examples")
    if resp:
        try:
            return resp.json()
        except json.JSONDecodeError:
            st.error("Invalid API response (rebuild_kg_profile_examples).")
            return None
    return None


def list_available_examples_modes(profile_name: str) -> Optional[Dict]:
    """Fetch the few-shot example modes available for a profile.

    Returns e.g. {"available": ["hardcoded", "template", "llm"], "default": "llm"},
    or None if the profile cannot be inspected.
    """
    if not profile_name:
        return None
    resp = safe_request("get", f"{API_BASE_URL}/profiles/{profile_name}/available_examples_modes")
    if resp:
        try:
            return resp.json()
        except json.JSONDecodeError:
            st.error(f"Invalid API response (list_available_examples_modes for {profile_name}).")
            return None
    return None

def delete_kg_profile(profile_name: str) -> bool:
    """Delete a KG profile via the API."""
    resp = safe_request("delete", f"{API_BASE_URL}/profiles/{profile_name}")
    return resp is not None

@st.cache_data(ttl=3600)
def get_label_color_mapping(profile_name: str) -> Optional[Dict[str, str]]:
    """Fetch the label-color mapping for a given KG profile from the API."""
    if not profile_name:
        st.warning("get_label_color_mapping called without a profile name.")
        return None

    endpoint_url = f"{API_BASE_URL}/profiles/{profile_name}/styles"
    resp = safe_request("get", endpoint_url)

    if resp:
        try:
            data = resp.json()
            if isinstance(data, dict):
                return data
            else:
                st.error(f"Unexpected API response for profile '{profile_name}' styles: not a dictionary.")
                st.json(data)
                return None
        except json.JSONDecodeError:
            st.error(f"Invalid JSON response for profile '{profile_name}' styles. Content: {resp.text}")
            return None
        except Exception as e:
            st.error(f"Error processing styles response for '{profile_name}': {e}")
            return None
    else:
        st.warning(f"Failed to fetch styles for profile '{profile_name}'.")
        return None


# ---------------------------------------------------------------------------
# flex result store
# ---------------------------------------------------------------------------


def fetch_result_by_id(result_id: str, limit: Optional[int] = None, offset: int = 0) -> Optional[Dict]:
    """Fetch a stored flex result. With `limit`, returns a paginated slice.

    Returns the parsed JSON dict, or None on error / not found.
    """
    if not result_id:
        return None
    endpoint_url = f"{API_BASE_URL}/results/{result_id}"
    params: Dict[str, object] = {}
    if limit is not None:
        params["limit"] = limit
        params["offset"] = offset
    try:
        resp = requests.get(endpoint_url, params=params, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.warning(f"Failed to fetch result {result_id}: {e}")
        return None


def fetch_thread_trace(thread_id: str) -> Optional[Dict]:
    """Fetch the flex trace rebuilt from a thread's checkpoint.

    Returns {thread_id, events, last_final}, or None on error.
    """
    if not thread_id:
        return None
    endpoint_url = f"{API_BASE_URL}/threads/{thread_id}/trace"
    try:
        resp = requests.get(endpoint_url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.warning(f"Failed to fetch trace for thread {thread_id}: {e}")
        return None
