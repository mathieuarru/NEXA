"""Storage for KG connection profiles and agent presets, as JSON files under KG_PROFILES_DIR.

A KG profile holds Neo4j credentials and per-profile settings (examples mode,
identifying properties, optional schema-description file). An agent preset holds
a system prompt and a structured-output schema.
"""

import os
import json
import shutil
import logging
from typing import List, Dict, Optional, Any, Tuple, Literal
from pydantic import BaseModel
from fastapi import HTTPException, UploadFile

from nexa.config import params
from nexa.models.dynamic_examples import (
    load_examples_for_profile as _load_examples_for_profile,
    load_identifying_properties_for_profile as _load_identifying_properties_for_profile,
)

logger = logging.getLogger(__name__)


############################
# Constants
############################
KG_PROFILES_DIR = params.KG_PROFILES_DIR
KG_PROFILES_JSON_PATH = os.path.join(KG_PROFILES_DIR, "kg_profiles.json")
KG_DESCRIPTIONS_DIR_PATH = os.path.join(KG_PROFILES_DIR, "kg_descriptions")
AGENT_PRESETS_JSON_PATH = os.path.join(KG_PROFILES_DIR, "agent_presets.json")


############################
# KG Profile Pydantic Models
############################
class KGProfileBase(BaseModel):
    """Base fields shared by KG connection profiles (host, port, credentials, and settings)."""
    host: str
    port: str
    username: str
    db_name: str
    browser_port: Optional[str] = "7474"
    examples_mode: Literal["hardcoded", "template", "llm"] = "hardcoded"
    identifying_properties: Optional[Dict[str, str]] = None

class KGProfileResponse(KGProfileBase):
    """A KG connection profile returned to clients, with its name and optional description file (password omitted)."""
    name: str
    description_file_name: Optional[str] = None

class AllKGProfilesResponse(BaseModel):
    """Response wrapper holding the list of all KG connection profiles."""
    profiles: List[KGProfileResponse]

############################
# Helper functions for KG Profile Management
############################
def load_profiles_from_disk() -> Dict[str, KGProfileResponse]:
    """Load all KG profiles from disk as name -> KGProfileResponse, stripping passwords.

    Returns an empty dict when the file is missing or cannot be parsed.
    """
    if not os.path.exists(KG_PROFILES_JSON_PATH):
        return {}
    try:
        with open(KG_PROFILES_JSON_PATH, "r") as f:
            profiles_data = json.load(f)
            loaded_profiles = {}
            for name, data in profiles_data.items():
                profile_data_for_response = data.copy()
                profile_data_for_response.pop("password", None)
                loaded_profiles[name] = KGProfileResponse(**profile_data_for_response)
            return loaded_profiles
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Error loading or parsing {KG_PROFILES_JSON_PATH}: {e}")
        return {}

def get_profile_with_password(profile_name: str) -> Optional[Dict[str, Any]]:
    """Return the raw data of one profile, password included, or None."""
    if not os.path.exists(KG_PROFILES_JSON_PATH):
        return None
    try:
        with open(KG_PROFILES_JSON_PATH, "r") as f:
            profiles_data = json.load(f)
            return profiles_data.get(profile_name)
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Error loading or parsing {KG_PROFILES_JSON_PATH} for password retrieval: {e}")
        return None

def save_profiles_to_disk(profiles: Dict[str, Any]):
    """Write the raw KG profiles mapping to disk as JSON, creating the profiles directory if needed."""
    os.makedirs(KG_PROFILES_DIR, exist_ok=True)
    try:
        with open(KG_PROFILES_JSON_PATH, "w") as f:
            json.dump(profiles, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving profiles to {KG_PROFILES_JSON_PATH}: {e}")
        raise HTTPException(status_code=500, detail="Could not save profiles to disk.")

def save_description_file(profile_name: str, file: UploadFile) -> str:
    """Save an uploaded schema-description file for a profile and return its sanitized file name."""
    os.makedirs(KG_DESCRIPTIONS_DIR_PATH, exist_ok=True)
    safe_profile_name = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in profile_name)
    file_name = f"{safe_profile_name}_description.json"
    file_path = os.path.join(KG_DESCRIPTIONS_DIR_PATH, file_name)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return file_name
    except Exception as e:
        logger.error(f"Could not save description file {file_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not save description file {file_name}.")
    finally:
        if hasattr(file, 'file') and file.file:
            file.file.close()

def delete_description_file(file_name: Optional[str]):
    """Delete a profile's description file from disk if it exists (no-op when the name is falsy)."""
    if file_name:
        file_path = os.path.join(KG_DESCRIPTIONS_DIR_PATH, file_name)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.error(f"Error deleting description file {file_path}: {e}")

def get_all_profiles_raw() -> Dict[str, Any]:
    """Return the raw profiles mapping, passwords included; empty on error or absence."""
    if not os.path.exists(KG_PROFILES_JSON_PATH):
        return {}
    try:
        with open(KG_PROFILES_JSON_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Error loading raw profiles from {KG_PROFILES_JSON_PATH}: {e}")
        return {}

def load_descriptions_from_profile(profile_data: Optional[Dict[str, Any]], profile_name_for_logging: str = "Unknown") -> Tuple[Optional[Dict], Optional[Dict]]:
    """Return (nodes_descriptions, relationships_descriptions) from the profile's description file.

    Either value is None when missing or unreadable.
    """
    if not profile_data:
        logger.info(f"No profile data provided for loading descriptions (profile: {profile_name_for_logging}).")
        return None, None

    description_file_name = profile_data.get("description_file_name")
    nodes_descriptions: Optional[Dict] = None
    relationships_descriptions: Optional[Dict] = None

    if description_file_name:
        file_path = os.path.join(KG_DESCRIPTIONS_DIR_PATH, description_file_name)
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    description_json_data = json.load(f)
                nodes_descriptions = description_json_data.get("nodes_descriptions")
                relationships_descriptions = description_json_data.get("relationships_descriptions")
                if nodes_descriptions or relationships_descriptions:
                    logger.info(f"Successfully loaded descriptions from {description_file_name} for profile '{profile_name_for_logging}'.")
                else:
                    logger.warning(f"Description file {description_file_name} for profile '{profile_name_for_logging}' exists but does not contain 'nodes_descriptions' or 'relationships_descriptions' keys.")
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from description file {file_path} for profile '{profile_name_for_logging}'.")
            except Exception as e:
                logger.error(f"Error reading description file {file_path} for profile '{profile_name_for_logging}': {e}")
        else:
            logger.warning(f"Description file {description_file_name} specified for profile '{profile_name_for_logging}' not found at {file_path}.")
    else:
        logger.info(f"No description file specified for KG profile '{profile_name_for_logging}'. Prompts will not use custom descriptions.")

    return nodes_descriptions, relationships_descriptions


def load_identifying_properties_from_description(profile_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    """Return the ``identifying_properties`` mapping from the profile's description file, or None."""
    if not profile_data:
        return None
    description_file_name = profile_data.get("description_file_name")
    if not description_file_name:
        return None
    file_path = os.path.join(KG_DESCRIPTIONS_DIR_PATH, description_file_name)
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        mapping = data.get("identifying_properties")
        if isinstance(mapping, dict):
            return {str(k): str(v) for k, v in mapping.items() if v}
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not read identifying_properties from {file_path}: {e}")
    return None


def load_dynamic_examples_for_profile(profile_name: str, examples_mode: str) -> Optional[Dict[str, str]]:
    """Return the cached few-shot examples for ``examples_mode``, or None (hardcoded mode, no cache, or error)."""
    if examples_mode == "hardcoded":
        return None
    try:
        return _load_examples_for_profile(profile_name, examples_mode)
    except Exception as e:
        logger.warning(f"Failed to load dynamic examples for profile '{profile_name}' (mode={examples_mode}): {e}")
        return None


def load_identifying_properties_for_profile(profile_name: str) -> Optional[Dict[str, str]]:
    """Return the identifying-property mapping resolved when the profile was built, or None."""
    try:
        return _load_identifying_properties_for_profile(profile_name)
    except Exception as e:
        logger.warning(f"Failed to load identifying properties for profile '{profile_name}': {e}")
        return None


def get_available_examples_modes(profile_name: str) -> Dict[str, Any]:
    """Return ``{"available": [...], "default": <profile examples_mode>}`` for a profile.

    "hardcoded" is always available; "template" and "llm" only when present in
    the examples cache. The cache's schema fingerprint is not checked.
    """
    from nexa.models.dynamic_examples import load_examples_from_disk

    available = ["hardcoded"]

    profiles_raw = get_all_profiles_raw()
    profile = profiles_raw.get(profile_name) or {}
    default_mode = profile.get("examples_mode", "hardcoded")

    payload = load_examples_from_disk(profile_name) or {}
    if payload.get("examples_template"):
        available.append("template")
    if payload.get("examples_llm"):
        available.append("llm")

    return {"available": available, "default": default_mode}


############################
# Agent Presets
############################
os.makedirs(KG_PROFILES_DIR, exist_ok=True)
os.makedirs(KG_DESCRIPTIONS_DIR_PATH, exist_ok=True)

class AgentPresetBase(BaseModel):
    """Base fields for an agent preset: an optional system prompt and structured-output schema."""
    system_prompt: Optional[str] = None
    structured_output_schema: Optional[str] = None

class AgentPresetResponse(AgentPresetBase):
    """An agent preset returned to clients, identified by name."""
    name: str

class AllAgentPresetsResponse(BaseModel):
    """Response wrapper holding the list of all agent presets."""
    presets: List[AgentPresetResponse]

def load_agent_presets_from_disk() -> Dict[str, AgentPresetResponse]:
    """Load all agent presets as name -> AgentPresetResponse."""
    if not os.path.exists(AGENT_PRESETS_JSON_PATH):
        return {}
    try:
        with open(AGENT_PRESETS_JSON_PATH, "r") as f:
            raw = json.load(f)
            result: Dict[str, AgentPresetResponse] = {}
            if isinstance(raw, dict):
                for name, data in raw.items():
                    result[name] = AgentPresetResponse(name=name,
                                                       system_prompt=data.get("system_prompt"),
                                                       structured_output_schema=data.get("structured_output_schema"))
            return result
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Error loading or parsing {AGENT_PRESETS_JSON_PATH}: {e}")
        return {}

def get_all_agent_presets_raw() -> Dict[str, Any]:
    """Load the raw agent presets mapping from disk. Returns an empty dict on error or absence."""
    if not os.path.exists(AGENT_PRESETS_JSON_PATH):
        return {}
    try:
        with open(AGENT_PRESETS_JSON_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Error loading raw presets from {AGENT_PRESETS_JSON_PATH}: {e}")
        return {}

def save_agent_presets_to_disk(presets: Dict[str, Any]):
    """Write the raw agent presets mapping to disk as JSON, creating the profiles directory if needed."""
    os.makedirs(KG_PROFILES_DIR, exist_ok=True)
    try:
        with open(AGENT_PRESETS_JSON_PATH, "w") as f:
            json.dump(presets, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving presets to {AGENT_PRESETS_JSON_PATH}: {e}")
        raise HTTPException(status_code=500, detail="Could not save agent presets to disk.")

def get_agent_preset_raw(name: str) -> Optional[Dict[str, Any]]:
    """Return the raw dict for a single agent preset by name, or None if not found."""
    raw = get_all_agent_presets_raw()
    return raw.get(name)
