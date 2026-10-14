"""Shared Streamlit UI: sidebar, logos, structured output, streaming and chat input."""

import base64
from pathlib import Path

import streamlit as st
import json
from nexa_frontend.api_utils import (
    get_ollama_models_from_api,
    get_openai_models_from_api,
    get_anthropic_models_from_api,
    get_gemini_models_from_api,
    get_azure_models_from_api,
    get_default_models,
    list_threads,
    create_new_thread,
    stream_message_to_thread,
    get_kg_profiles,
    list_available_examples_modes,
)
from nexa_frontend.streamlit_callback import StreamlitCallback
import re

_LOGO_DIR = Path(__file__).parent / "nexa_logo"


@st.cache_data(show_spinner=False)
def _logo_data_uri(filename: str) -> str:
    """Return a base64 data URI for a logo file (cached across reruns)."""
    encoded = base64.b64encode((_LOGO_DIR / filename).read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _is_dark_theme() -> bool:
    """Whether Streamlit's active theme is dark (light if unknown).

    Read from `st.context.theme`, which only refreshes on a rerun.
    """
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def assistant_avatar() -> str:
    """Return the path of the assistant avatar image matching the current theme."""
    filename = "NEXA_network_white_crop.png" if _is_dark_theme() else "NEXA_network_crop.png"
    return str(_LOGO_DIR / filename)


def _render_sidebar_logo():
    """Render the NEXA wordmark, without tagline, at the top of the sidebar."""
    logo_file = "NEXA_white_crop.png" if _is_dark_theme() else "NEXA_black_crop.png"
    # The `_crop` assets are tightly cropped, so these margins are the visible
    # gaps; the negative top margin offsets the sidebar's default padding.
    _pull_up = "-2.75rem"
    _gap_below = "1.5rem"
    st.markdown(
        f"""
        <div style="display:flex; justify-content:center; margin:{_pull_up} 0 {_gap_below} 0;">
            <img src="{_logo_data_uri(logo_file)}" alt="NEXA"
                 style="width:90%; max-width:260px; height:auto;" />
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_welcome_logo():
    """Render the full NEXA wordmark, centered, on a new conversation."""
    logo_file = "NEXA_white.png" if _is_dark_theme() else "NEXA_black.png"
    st.markdown(
        f"""
        <div style="display:flex; justify-content:center; margin:3rem 0 1.5rem 0;">
            <img src="{_logo_data_uri(logo_file)}" alt="NEXA"
                 style="width:65%; max-width:540px; min-width:280px; height:auto;" />
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render the sidebar and return whether a usable model is selected."""
    with st.sidebar:

        _render_sidebar_logo()
        if st.button("New Conversation", key="new_conversation_sidebar", width="stretch", icon="🆕"):
            st.session_state.current_thread_id = None
            st.session_state.data = {}
            st.session_state.just_created_thread = False
            st.session_state.just_posted_message = False
            st.session_state.last_user_message = ""
            st.session_state.title = ""
            if "last_callback_history" in st.session_state:
                del st.session_state.last_callback_history
            st.session_state.current_page = "chat"
            # Restore the last mode chosen explicitly (opening a thread may
            # have switched it).
            _restored = st.session_state.get("last_user_chosen_mode", "direct")
            if _restored in ("direct", "flex"):
                st.session_state.selected_agent_mode = _restored
                # The widget key is synced on the next rerun, before the
                # widget is created.
                st.session_state["_pending_mode_widget_sync"] = True
            st.rerun()

        # Agent architecture switch; the explanation is in the `help` tooltip.
        _agent_modes = {"Direct": "direct", "Flex": "flex"}
        _help_text = (
            "**Direct** — deterministic pipeline (decompose → generate → "
            "correct Cypher). Single-shot and fast; best for one-hop "
            "questions or high-volume benchmarking. Works well with small "
            "models.\n\n"
            "**Flex** — autonomous agent with granular tools "
            "(`execute_cypher`, `lookup_property`, `filter_result`, "
            "`aggregate_result`, …). Multi-step reasoning with full "
            "traceability. Requires a capable LLM (Claude Haiku, GPT-4o)."
        )
        _current_agent_mode = st.session_state.get("selected_agent_mode", "direct")
        # "pipeline" is an alias of the direct mode.
        if _current_agent_mode == "pipeline":
            _current_agent_mode = "direct"
            st.session_state.selected_agent_mode = "direct"
        _current_label = next(
            (label for label, value in _agent_modes.items() if value == _current_agent_mode),
            "Direct",
        )

        # Split the segmented control's options evenly across the sidebar.
        st.markdown(
            """
            <style>
            div[data-testid="stSegmentedControl"] > div {
                width: 100%;
            }
            div[data-testid="stSegmentedControl"] label {
                flex: 1 1 0;
                text-align: center;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Sync the widget key from selected_agent_mode before the widget is
        # created (Streamlit forbids writing it afterwards): on first render,
        # and after a mode switch flagged by `_pending_mode_widget_sync`.
        _segmented_key = "agent_mode_segmented_sidebar"
        _radio_key = "agent_mode_radio_sidebar"
        _segmented = getattr(st, "segmented_control", None) or getattr(st, "pills", None)
        _active_widget_key = _segmented_key if _segmented is not None else _radio_key
        _needs_first_prime = _active_widget_key not in st.session_state
        _has_pending_sync = st.session_state.pop("_pending_mode_widget_sync", False)
        if _needs_first_prime or _has_pending_sync:
            st.session_state[_active_widget_key] = _current_label

        # A thread keeps the mode it was created in; switching requires a new
        # conversation.
        _in_active_thread = bool(st.session_state.get("current_thread_id"))
        _effective_help = _help_text if not _in_active_thread else (
            "**Mode is locked to the conversation's original mode.** "
            "Click **New Conversation** to switch."
        )
        if _segmented is not None:
            _sel_label = _segmented(
                "Mode",
                options=list(_agent_modes.keys()),
                key=_segmented_key,
                width="stretch",
                help=_effective_help,
                disabled=_in_active_thread,
            )
        else:
            # Streamlit versions without segmented_control.
            _sel_label = st.radio(
                "Mode",
                options=list(_agent_modes.keys()),
                horizontal=True,
                key=_radio_key,
                help=_effective_help,
                disabled=_in_active_thread,
            )
        # Deselecting the active option returns None; keep the current mode.
        if not _sel_label:
            _sel_label = _current_label
        _new_selected_mode = _agent_modes[_sel_label]
        # A difference here means the user changed the mode; record the choice.
        if _new_selected_mode != st.session_state.selected_agent_mode:
            st.session_state.selected_agent_mode = _new_selected_mode
            st.session_state.last_user_chosen_mode = _new_selected_mode
        # Options with no effect on the flex agent are hidden in flex mode.
        is_flex = st.session_state.selected_agent_mode == "flex"

        with st.expander("Agent Configuration", expanded=False):

            st.subheader("LLM Configuration")
            providers = ["ollama", "openai", "anthropic", "gemini", "azure"]
            try:
                prov_index = providers.index(st.session_state.selected_provider)
            except ValueError:
                prov_index = 0

            new_provider = st.selectbox(
                "LLM Provider:",
                providers,
                index=prov_index,
                key="provider_selector_sidebar"
            )

            if new_provider != st.session_state.selected_provider:
                st.session_state.selected_provider = new_provider
                st.session_state.selected_model_name = ""
                st.rerun()

            available_models = []
            is_selectable = False

            openai_key = st.session_state.get("openai_api_key_stored", "")
            anthropic_key = st.session_state.get("anthropic_api_key_stored", "")
            google_key = st.session_state.get("google_api_key_stored", "")
            azure_key = st.session_state.get("azure_api_key_stored", "")
            azure_endpoint = st.session_state.get("azure_endpoint_stored", "")

            # Default models come from the API's environment configuration.
            defaults = get_default_models()
            provider = st.session_state.selected_provider
            default_model = defaults.get(provider, "")

            if provider == "ollama":
                available_models = get_ollama_models_from_api()
                if not available_models:
                    st.warning("Unable to fetch Ollama models from the API.")
                    available_models = ["API Error"]
            elif provider == "openai":
                available_models = get_openai_models_from_api(api_key=openai_key)
                if not available_models:
                     st.warning("Unable to fetch OpenAI models. Please configure your API key in Settings > API Keys.")
                     available_models = ["API Error"]
            elif provider == "anthropic":
                available_models = get_anthropic_models_from_api(api_key=anthropic_key)
                if not available_models:
                    st.warning("Unable to fetch Anthropic models. Please configure your API key in Settings > API Keys.")
                    available_models = ["API Error"]
            elif provider == "gemini":
                available_models = get_gemini_models_from_api(api_key=google_key)
                if not available_models:
                    st.warning("Unable to fetch Gemini models. Please configure your API key in Settings > API Keys.")
                    available_models = ["API Error"]
            elif provider == "azure":
                available_models = get_azure_models_from_api(api_key=azure_key, endpoint=azure_endpoint)
                if not available_models:
                    st.warning("Unable to fetch Azure deployments. Please configure your API key in Settings > API Keys.")
                    available_models = ["API Error"]
            else:
                available_models = ["N/A"]
                default_model = "N/A"

            # If no default configured, use first available model
            if not default_model and available_models and available_models[0] not in ["N/A", "API Error"]:
                 st.session_state.selected_model_name = available_models[0]
                 default_model = available_models[0]

            current_model_valid = (
                st.session_state.get("selected_model_name") and
                st.session_state.selected_model_name in available_models and
                st.session_state.selected_model_name not in ["N/A", "API Error"]
            )

            if not current_model_valid:
                 if default_model and default_model not in ["N/A", "API Error"]:
                      st.session_state.selected_model_name = default_model
                 else:
                      st.session_state.selected_model_name = ""

            model_index = 0
            if st.session_state.selected_model_name and st.session_state.selected_model_name in available_models:
                try:
                    model_index = available_models.index(st.session_state.selected_model_name)
                except ValueError:
                    if default_model and default_model in available_models:
                         try:
                             model_index = available_models.index(default_model)
                             st.session_state.selected_model_name = default_model
                         except ValueError:
                             model_index = 0
                             st.session_state.selected_model_name = ""
                    else:
                        model_index = 0
                        st.session_state.selected_model_name = ""

            is_selectable = bool(available_models and available_models[0] not in ["N/A", "API Error"])

            new_model_name = st.selectbox(
                f"Model ({st.session_state.selected_provider}):",
                available_models,
                index=model_index,
                key="model_name_selector_sidebar",
                disabled=not is_selectable
            )

            if new_model_name != st.session_state.selected_model_name and is_selectable:
                st.session_state.selected_model_name = new_model_name

            st.subheader("Knowledge Graph")

            profiles = get_kg_profiles()
            if not profiles:
                st.warning("No KG profiles available. Please create one in Settings.")
                return

            profile_names = [p["name"] for p in profiles]
            selected_profile = st.selectbox(
                "Select KG Profile:",
                options=profile_names,
                index=profile_names.index(st.session_state.selected_kg_profile_name) if st.session_state.selected_kg_profile_name in profile_names else 0,
                key="kg_profile_selector_sidebar"
            )

            if selected_profile != st.session_state.selected_kg_profile_name:
                st.session_state.selected_kg_profile_name = selected_profile
                st.session_state.selected_examples_mode_override = None
                st.rerun()

            # KG mode applies to the direct pipeline only.
            if not is_flex:
                kg_modes = ["Off", "Auto", "Force"]
                if 'kg_mode' not in st.session_state or st.session_state.kg_mode not in kg_modes:
                    st.session_state.kg_mode = "Auto"

                selected_kg_mode = st.radio(
                    "KG Mode:",
                    options=kg_modes,
                    index=kg_modes.index(st.session_state.kg_mode),
                    key="kg_mode_selector_sidebar",
                    horizontal=True,
                    help="Defines how the agent uses the Knowledge Graph..."
                )
                if selected_kg_mode != st.session_state.kg_mode:
                    st.session_state.kg_mode = selected_kg_mode

            st.subheader("Prompt & Structured Output")

            from nexa_frontend.api_utils import list_agent_presets
            presets = list_agent_presets() or []

            prompt_names = [p["name"] for p in presets if isinstance(p, dict) and p.get("system_prompt")]
            schema_names = [p["name"] for p in presets if isinstance(p, dict) and p.get("structured_output_schema")]

            prompt_options = ["Default"] + prompt_names
            current_prompt_name = st.session_state.get("selected_system_prompt_name")
            if current_prompt_name in prompt_names:
                prompt_index = 1 + prompt_names.index(current_prompt_name)
            else:
                prompt_index = 0
            sel_prompt_label = st.selectbox(
                "System Prompt",
                options=prompt_options,
                index=prompt_index,
                key="selected_system_prompt_name_selector_sidebar",
                help="Built-in system prompt."
            )
            st.session_state.selected_system_prompt_name = None if sel_prompt_label == "Default" else sel_prompt_label

            schema_options = ["None"] + schema_names
            current_schema_name = st.session_state.get("selected_structured_output_name")
            if current_schema_name in schema_names:
                schema_index = 1 + schema_names.index(current_schema_name)
            else:
                schema_index = 0
            sel_schema_label = st.selectbox(
                "Structured Output",
                options=schema_options,
                index=schema_index,
                key="selected_structured_output_name_selector_sidebar",
                help="'None' disables the structured output."
            )
            st.session_state.selected_structured_output_name = None if sel_schema_label == "None" else sel_schema_label

            # Few-shot examples and Cypher generation apply to the direct
            # pipeline only.
            if not is_flex:
                modes_info = list_available_examples_modes(st.session_state.selected_kg_profile_name) or {}
                available_modes = modes_info.get("available") or ["hardcoded"]
                profile_default_mode = modes_info.get("default") or "hardcoded"
                available_mode_labels = [m.capitalize() for m in available_modes]
                examples_options = [f"Default - {profile_default_mode.capitalize()}"] + available_mode_labels

                current_override = st.session_state.get("selected_examples_mode_override")
                if current_override in available_modes:
                    examples_index = 1 + available_modes.index(current_override)
                else:
                    examples_index = 0
                    if current_override is not None:
                        st.session_state.selected_examples_mode_override = None

                sel_examples_label = st.selectbox(
                    "Few-shot Examples",
                    options=examples_options,
                    index=examples_index,
                    key="selected_examples_mode_selector_sidebar",
                    help=(
                        "Overrides the profile's few-shot examples mode for this session. "
                        "Only modes with a cached build are listed; rebuild a profile in Settings to enable others."
                    ),
                )
                if sel_examples_label.startswith("Default -"):
                    st.session_state.selected_examples_mode_override = None
                else:
                    st.session_state.selected_examples_mode_override = sel_examples_label.lower()

                generation_modes = {
                    "Decomposed": "decomposed",
                    "One-shot (simple)": "one_shot",
                    "One-shot + fallback decomposed": "one_shot_fallback",
                }
                current_gen_mode = st.session_state.get("selected_generation_mode", "one_shot_fallback")
                current_gen_label = next(
                    (label for label, value in generation_modes.items() if value == current_gen_mode),
                    "One-shot + fallback decomposed",
                )
                sel_gen_label = st.selectbox(
                    "Cypher Generation",
                    options=list(generation_modes.keys()),
                    index=list(generation_modes.keys()).index(current_gen_label),
                    key="selected_generation_mode_selector_sidebar",
                    help=(
                        "Decomposed: full NEXA pipeline (rephrase → label select → rel select → gen → correct). "
                        "One-shot: a single LLM call with the full schema (GraphCypherQA-style). "
                        "One-shot + fallback: try one-shot first, fall back to decomposed if no data is returned."
                    ),
                )
                st.session_state.selected_generation_mode = generation_modes[sel_gen_label]


        with st.expander("Conversation History", expanded=True):
            threads = list_threads()
            if not threads:
                st.write("No conversations found.")
            else:
                for t in sorted(threads, key=lambda x: x.get('title', f"Thread {x.get('thread_id', 'NoID')[:8]}").lower()):
                    title_display = t.get('title', f"Thread {t.get('thread_id', 'NoID')[:8]}")
                    if st.button(title_display, key=f"thread_sidebar_{t['thread_id']}", width="stretch"):
                        st.session_state.current_thread_id = t["thread_id"]
                        st.session_state.just_created_thread = False
                        st.session_state.just_posted_message = False
                        st.session_state.last_user_message = ""
                        if "last_callback_history" in st.session_state:
                            del st.session_state.last_callback_history
                        st.session_state.current_page = "chat"
                        # Switch to the thread's own mode ("direct" for
                        # threads without one).
                        _thread_mode = t.get("agent_mode") or "direct"
                        if _thread_mode in ("direct", "flex"):
                            st.session_state.selected_agent_mode = _thread_mode
                            # The widget key is synced on the next rerun.
                            st.session_state["_pending_mode_widget_sync"] = True
                        st.rerun()

        if st.button("Application Settings", key="go_to_settings_page_button_sidebar", width="stretch", icon="⚙️"):
            st.session_state.current_page = "settings"
            st.rerun()

    model_ready = st.session_state.get("selected_model_name") and is_selectable and st.session_state.selected_model_name not in ["N/A", "API Error"]
    return model_ready

def parse_thinking_content(content: str) -> tuple[str, str]:
    """Split `<think>...</think>` blocks from a message; return (thinking, message)."""
    thinking_content = ""
    message_content = content

    think_pattern = r'<think>(.*?)</think>'
    matches = re.findall(think_pattern, content, re.DOTALL)

    if matches:
        thinking_content = " ".join(matches).strip()
        message_content = re.sub(think_pattern, '', content, flags=re.DOTALL).strip()

    return thinking_content, message_content

def format_structured_output(content: str) -> str | None:
    """Format a JSON object as Markdown, one entry per key.

    Returns None if `content` is not a non-empty JSON object.
    """
    try:
        parsed = json.loads(content)
        if not isinstance(parsed, dict) or not parsed:
            return None
    except (json.JSONDecodeError, TypeError):
        return None

    lines = []
    for key, value in parsed.items():
        display_key = key.replace("_", " ").capitalize()

        if isinstance(value, list):
            lines.append(f"**{display_key}:**")
            for item in value:
                lines.append(f"- {item}")
        elif isinstance(value, bool):
            lines.append(f"**{display_key}:** {'✅ True' if value else '❌ False'}")
        elif isinstance(value, dict):
            lines.append(f"**{display_key}:** `{json.dumps(value)}`")
        elif value is None:
            lines.append(f"**{display_key}:** _None_")
        else:
            lines.append(f"**{display_key}:** {value}")

        lines.append("")

    return "\n".join(lines).strip()

def handle_streaming(is_model_ready: bool):
    """Show the pending user message, stream the assistant reply, then rerun."""
    callback_instance = None
    final_history = None

    # Flex: a placeholder between the user and assistant bubbles, where
    # StreamlitCallback re-renders the live trace on each tool event.
    _is_flex = st.session_state.get("selected_agent_mode") == "flex"

    def _install_flex_live_trace_placeholder():
        if not _is_flex:
            return
        ph = st.empty()
        st.session_state["_flex_trace_placeholder"] = ph

    if st.session_state.just_created_thread and st.session_state.current_thread_id:
        if st.session_state.get("title"):
             st.title(f"{st.session_state.title}")
        with st.chat_message("user"):
            st.markdown(st.session_state.last_user_message)
        _install_flex_live_trace_placeholder()
        # Inject the Stop button before write_stream blocks the script.
        from nexa_frontend.streaming_control import render_stop_button
        render_stop_button(st.session_state.current_thread_id)
        with st.chat_message("assistant", avatar=assistant_avatar()):
            with st.spinner("Thinking..."):
                if is_model_ready:
                    callback_container = st.container()
                    callback_instance = StreamlitCallback(callback_container)
                    st.write_stream(stream_message_to_thread(
                        user_message=st.session_state.last_user_message,
                        thread_id=st.session_state.current_thread_id,
                        title=st.session_state.title,
                        model_provider=st.session_state.selected_provider,
                        model_name=st.session_state.selected_model_name,
                        callback=callback_instance
                    ))
                    callback_instance.complete()
                    final_history = callback_instance.get_final_history()
                else:
                    st.error("No valid model selected. Unable to start.")

                if final_history:
                     st.session_state.last_callback_history = final_history
                st.session_state.just_created_thread = False
                st.session_state.last_user_message = ""
                thread_id_to_clear = st.session_state.current_thread_id
                if thread_id_to_clear in st.session_state.get('history_cache', {}):
                    del st.session_state.history_cache[thread_id_to_clear]

    elif st.session_state.just_posted_message and st.session_state.current_thread_id:
        with st.chat_message("user"):
            st.markdown(st.session_state.last_user_message)
        _install_flex_live_trace_placeholder()
        from nexa_frontend.streaming_control import render_stop_button
        render_stop_button(st.session_state.current_thread_id)
        with st.chat_message("assistant", avatar=assistant_avatar()):
            with st.spinner("Thinking..."):
                 if is_model_ready:
                    callback_container = st.container()
                    callback_instance = StreamlitCallback(callback_container)
                    st.write_stream(stream_message_to_thread(
                        thread_id=st.session_state.current_thread_id,
                        user_message=st.session_state.last_user_message,
                        model_provider=st.session_state.selected_provider,
                        model_name=st.session_state.selected_model_name,
                        callback=callback_instance
                    ))
                    callback_instance.complete()
                    final_history = callback_instance.get_final_history()
                 else:
                    st.error("No valid model selected. Unable to continue.")

                 if final_history:
                      st.session_state.last_callback_history = final_history
                 st.session_state.just_posted_message = False
                 st.session_state.last_user_message = ""
                 thread_id_to_clear = st.session_state.current_thread_id
                 if thread_id_to_clear in st.session_state.get('history_cache', {}):
                     del st.session_state.history_cache[thread_id_to_clear]

    if callback_instance is not None:
         # Flex: reload the persisted trace after the rerun; live events have
         # no `turn` and would not be shown in the history.
         if _is_flex:
             from nexa_frontend.chat_flex.state import invalidate_trace_hydration
             invalidate_trace_hydration()
         st.rerun()

def render_user_input(is_model_ready: bool, disabled: bool = False):
    """Render the chat input and start a new thread or post the message.

    Kept visible while streaming (`disabled=True`), with the Stop overlay
    replacing its send button.
    """
    if is_model_ready:
        placeholder = "Enter your message..."
        if st.session_state.current_thread_id is None:
            placeholder = "Enter your message to start a new conversation..."
    else:
        placeholder = "Please select a valid provider and model."

    user_input = st.chat_input(
        placeholder,
        disabled=(not is_model_ready) or disabled,
        key="main_chat_input",
    )

    if user_input and is_model_ready:
        with st.spinner("Loading models..."):
            if st.session_state.current_thread_id is None:
                thread_info = create_new_thread(
                    user_input,
                    st.session_state.selected_provider,
                    st.session_state.selected_model_name
                )
                if thread_info and "thread_id" in thread_info:
                    st.session_state.current_thread_id = thread_info["thread_id"]
                    st.session_state.title = thread_info.get("title", "Conversation")
                    st.session_state.last_user_message = user_input
                    st.session_state.just_created_thread = True
                    st.rerun()
                else:
                    st.error("Unable to create a new conversation.")
            else:
                 st.session_state.last_user_message = user_input
                 st.session_state.just_posted_message = True
                 st.rerun()

