"""Default Streamlit session-state keys."""

import streamlit as st


def initialize_session_state():
    """Initialize required keys in st.session_state if they do not exist."""

    defaults = {
        "current_thread_id": None,
        "data": {},
        "just_created_thread": False,
        "just_posted_message": False,
        "last_user_message": "",
        "title": "",
        "selected_provider": "openai",
        "selected_model_name": "",
        "current_page": "chat",
        "selected_kg_profile_name": None,
        "openai_api_key_stored": "",
        "anthropic_api_key_stored": "",
        "google_api_key_stored": "",
        "azure_api_key_stored": "",
        "azure_endpoint_stored": "",
        "last_callback_history": None,
        "history_cache": {},
        # Active architecture. Flex-only keys are set by chat_flex/state.py.
        "selected_agent_mode": "direct",
        # Last mode chosen explicitly in the sidebar (opening a thread may
        # switch `selected_agent_mode`); restored on "New Conversation".
        "last_user_chosen_mode": "direct",
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
