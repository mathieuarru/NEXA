"""NEXA Streamlit frontend entry point.

Renders the settings page or the Direct or Flex chat page, depending on
`st.session_state.selected_agent_mode`.
"""

import streamlit as st

from nexa_frontend.chat_direct import render_chat_page_direct
from nexa_frontend.chat_flex import render_chat_page_flex
from nexa_frontend.settings_page import render_settings_page
from nexa_frontend.state_manager import initialize_session_state
from nexa_frontend.ui_components import render_sidebar


def main():
    st.set_page_config(layout="wide", page_title="NEXA")

    initialize_session_state()

    is_model_ready = render_sidebar()

    if st.session_state.get("current_page") == "settings":
        render_settings_page()
        return

    mode = st.session_state.get("selected_agent_mode", "direct")
    # "pipeline" is an alias of the direct mode.
    if mode == "pipeline":
        mode = "direct"
    if mode == "flex":
        render_chat_page_flex(is_model_ready)
    else:
        render_chat_page_direct(is_model_ready)


if __name__ == "__main__":
    main()
