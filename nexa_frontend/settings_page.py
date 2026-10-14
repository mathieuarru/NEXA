"""Settings page: conversations, KG profiles, agent presets and API keys."""

import json
import streamlit as st
from nexa_frontend.api_utils import (
    get_kg_profiles,
    get_kg_profile,
    create_kg_profile,
    update_kg_profile,
    rebuild_kg_profile_examples,
    delete_kg_profile,
    delete_all_threads,
    list_agent_presets,
    create_agent_preset,
    update_agent_preset,
    delete_agent_preset
)

def render_settings_page():
    st.header("⚙️ Application Settings")

    tab_general, tab_kg, tab_agent_presets, tab_api_keys = st.tabs(["General", "KG Profiles", "Agent Presets", "API Keys"])

    with tab_general:
        render_general_settings_tab()

    with tab_kg:
        render_kg_profiles_tab()

    with tab_agent_presets:
        render_agent_presets_tab()

    with tab_api_keys:
        render_api_keys_tab()

    st.markdown("---")
    if st.button("Back to Chat", key="back_to_chat_button_main_settings"):
        st.session_state.current_page = "chat"
        st.rerun()

def render_agent_presets_tab():
    st.subheader("Agent Presets: System Prompts & Structured Outputs (server-side)")


    presets_list = list_agent_presets() or []
    presets_by_name = {p.get("name"): p for p in presets_list if isinstance(p, dict) and p.get("name")}

    ############################
    # SYSTEM PROMPTS
    ############################
    st.markdown("#### System Prompts")
    sp_names = [name for name, p in presets_by_name.items() if p.get("system_prompt")]
    CREATE_NEW_PROMPT_LABEL = "➕ Create new prompt"
    sp_options = [CREATE_NEW_PROMPT_LABEL] + sp_names
    selected_sp_label = st.selectbox(
        "Select a prompt to view/edit:", options=sp_options, key="agent_preset_prompt_selector"
    )
    is_creating_prompt = selected_sp_label == CREATE_NEW_PROMPT_LABEL
    selected_sp_name = None if is_creating_prompt else selected_sp_label

    current_sp_value = presets_by_name.get(selected_sp_name, {}).get("system_prompt", "") if selected_sp_name else ""
    sp_new_name = st.text_input(
        "Name:",
        value=(selected_sp_name or ""),
        key="agent_preset_prompt_name",
        placeholder="E.g.: My custom prompt"
    )
    sp_new_content = st.text_area(
        "Content:",
        value=(current_sp_value if not is_creating_prompt else ""),
        height=180,
        key="agent_preset_prompt_content",
        placeholder="E.g.: Edit this field to modify how NEXA interprets queries, reasons over the knowledge graph, and formats its answers. "
    )

    colp1, colp2, colp3 = st.columns(3)
    with colp1:
        if st.button("Save New", key="save_new_prompt_button"):
            if sp_new_name.strip() and sp_new_content.strip():
                result = create_agent_preset(sp_new_name.strip(), system_prompt=sp_new_content, structured_output_schema=None)
                if result:
                    st.success("Prompt saved.")
                    st.rerun()
                else:
                    st.error("Failed to save prompt.")
            else:
                st.error("Name and content cannot be empty.")
    with colp2:
        disabled_update = is_creating_prompt or not selected_sp_name
        if st.button("Update Selected", key="update_selected_prompt_button", disabled=disabled_update):
            result = update_agent_preset(selected_sp_name, system_prompt=sp_new_content)
            if result:
                st.success("Prompt updated.")
                st.rerun()
            else:
                st.error("Failed to update prompt.")
    with colp3:
        disabled_delete = is_creating_prompt or not selected_sp_name
        if st.button("Delete Selected", key="delete_selected_prompt_button", disabled=disabled_delete):
            if delete_agent_preset(selected_sp_name):
                st.warning("Prompt deleted.")
                st.rerun()
            else:
                st.error("Failed to delete prompt.")

    st.markdown("---")

    ############################
    # STRUCTURED OUTPUT SCHEMAS
    ############################
    st.markdown("#### Structured Output Schemas")
    so_names = [name for name, p in presets_by_name.items() if p.get("structured_output_schema")]
    CREATE_NEW_SCHEMA_LABEL = "➕ Create new structured output"
    so_options = [CREATE_NEW_SCHEMA_LABEL] + so_names
    selected_so_label = st.selectbox(
        "Select a schema to view/edit:", options=so_options, key="agent_preset_schema_selector"
    )
    is_creating_schema = selected_so_label == CREATE_NEW_SCHEMA_LABEL
    selected_so_name = None if is_creating_schema else selected_so_label

    current_so_value = presets_by_name.get(selected_so_name, {}).get("structured_output_schema", "") if selected_so_name else ""
    so_new_name = st.text_input(
        "Name:",
        value=(selected_so_name or ""),
        key="agent_preset_schema_name",
        placeholder="E.g.: summary_schema"
    )
    so_new_content = st.text_area(
        "Schema (JSON):",
        value=(current_so_value if not is_creating_schema else ""),
        height=220,
        key="agent_preset_schema_content",
        placeholder="{\n  \"type\": \"object\",\n  \"properties\": {\n    \"short_answer\": { \"type\": \"string\" },\n    \"key_points\": { \"type\": \"array\", \"items\": { \"type\": \"string\" } }\n  },\n  \"required\": [\"short_answer\"]\n}"
    )

    cols1, cols2, cols3 = st.columns(3)
    with cols1:
        if st.button("Save New", key="save_new_schema_button"):
            if so_new_name.strip() and so_new_content.strip():
                result = create_agent_preset(so_new_name.strip(), system_prompt=None, structured_output_schema=so_new_content)
                if result:
                    st.success("Schema saved.")
                    st.rerun()
                else:
                    st.error("Failed to save schema.")
            else:
                st.error("Name and schema cannot be empty.")
    with cols2:
        disabled_update_so = is_creating_schema or not selected_so_name
        if st.button("Update Selected", key="update_selected_schema_button", disabled=disabled_update_so):
            result = update_agent_preset(selected_so_name, structured_output_schema=so_new_content)
            if result:
                st.success("Schema updated.")
                st.rerun()
            else:
                st.error("Failed to update schema.")
    with cols3:
        disabled_delete_so = is_creating_schema or not selected_so_name
        if st.button("Delete Selected", key="delete_selected_schema_button", disabled=disabled_delete_so):
            if delete_agent_preset(selected_so_name):
                st.warning("Schema deleted.")
                st.rerun()
            else:
                st.error("Failed to delete schema.")

def render_kg_profiles_tab():
    st.subheader("Knowledge Graph Connection Profiles")

    ############################
    # PROFILE MANAGEMENT
    ############################
    action = st.selectbox("Action:", ["View/Edit Profiles", "Create New Profile"], key="profile_action_selector_kg")

    if 'editing_profile_name' not in st.session_state:
        st.session_state.editing_profile_name = None

    if action == "Create New Profile":
        st.markdown("##### Create New KG Profile")
        new_profile_name = st.text_input("New Profile Name:", key="new_profile_name_input_kg").strip()
        new_host = st.text_input("Neo4j Host:", value="localhost", key="new_host_input_kg")
        new_port = st.text_input("Neo4j Bolt Port:", value="7687", key="new_port_input_kg")
        new_user = st.text_input("Neo4j Username:", value="neo4j", key="new_user_input_kg")
        new_password = st.text_input("Neo4j Password:", value="password", type="password", key="new_password_input_kg")
        new_db_name = st.text_input("Neo4j Database Name:", value="neo4j", key="new_db_name_input_kg")
        new_browser_port = st.text_input("Neo4j Browser Port (HTTP):", value="7474", key="new_browser_port_input_kg")
        new_desc_file = st.file_uploader("Upload KG Description JSON (Optional)", type=['json'], key="new_desc_file_uploader_kg")

        st.markdown("**Few-shot examples (Advanced)**")
        new_examples_mode = st.selectbox(
            "Few-shot examples mode:",
            options=["hardcoded", "template", "llm"],
            index=0,
            help=(
                "hardcoded: shipped defaults (Hetionet-flavoured). "
                "template: deterministic examples generated from this graph's schema. "
                "llm: same Cypher, natural-language questions authored by an LLM."
            ),
            key="new_examples_mode_input_kg",
        )
        new_identifying_properties_raw = st.text_area(
            "Identifying properties (optional JSON mapping label → property):",
            value="",
            placeholder='{"Compound": "name", "Gene": "geneSymbol"}',
            help="Tells the example builder which property a human would use to name an entity of each label.",
            key="new_identifying_properties_input_kg",
        ).strip()

        if st.button("Save New Profile", key="save_new_profile_button_kg"):
            if not new_profile_name:
                st.error("Profile Name cannot be empty.")
            else:
                description_file_content = None
                if new_desc_file:
                    description_file_content = new_desc_file.getvalue()

                result = create_kg_profile(
                    name=new_profile_name,
                    host=new_host,
                    port=new_port,
                    username=new_user,
                    password=new_password,
                    db_name=new_db_name,
                    browser_port=new_browser_port,
                    description_file=description_file_content,
                    examples_mode=new_examples_mode,
                    identifying_properties=new_identifying_properties_raw or None,
                )

                if result:
                    st.success(f"Profile '{new_profile_name}' created successfully!")
                    st.session_state.selected_kg_profile_name = new_profile_name
                    st.rerun()
                else:
                    st.error("Failed to create profile. Please check the API logs.")

    elif action == "View/Edit Profiles":
        st.markdown("##### Manage Existing KG Profiles")
        profiles = get_kg_profiles()
        if not profiles:
            st.info("No profiles created yet. Please create a new profile.")
            return

        profile_names = [p["name"] for p in profiles]

        if st.session_state.editing_profile_name is None and profile_names:
            if st.session_state.selected_kg_profile_name in profile_names:
                st.session_state.editing_profile_name = st.session_state.selected_kg_profile_name
            else:
                st.session_state.editing_profile_name = profile_names[0]
        elif st.session_state.editing_profile_name and st.session_state.editing_profile_name not in profile_names:
            st.session_state.editing_profile_name = profile_names[0] if profile_names else None

        selected_for_editing = st.selectbox(
            "Select Profile to View/Edit:",
            options=profile_names,
            index=profile_names.index(st.session_state.editing_profile_name) if st.session_state.editing_profile_name and st.session_state.editing_profile_name in profile_names else 0,
            key="edit_profile_selector_kg"
        )

        if selected_for_editing != st.session_state.editing_profile_name:
            st.session_state.editing_profile_name = selected_for_editing
            st.rerun()

        if st.session_state.editing_profile_name:
            profile_data = get_kg_profile(st.session_state.editing_profile_name)
            if profile_data:
                st.markdown(f"###### Editing: {st.session_state.editing_profile_name}")

                edit_host = st.text_input("Neo4j Host:", value=profile_data["host"], key=f"edit_host_kg_{st.session_state.editing_profile_name}")
                edit_port = st.text_input("Neo4j Bolt Port:", value=profile_data["port"], key=f"edit_port_kg_{st.session_state.editing_profile_name}")
                edit_user = st.text_input("Neo4j Username:", value=profile_data["username"], key=f"edit_user_kg_{st.session_state.editing_profile_name}")
                edit_password = st.text_input("Neo4j Password:", value="password", type="password", key=f"edit_password_kg_{st.session_state.editing_profile_name}")
                edit_db_name = st.text_input("Neo4j Database Name:", value=profile_data["db_name"], key=f"edit_db_name_kg_{st.session_state.editing_profile_name}")
                edit_browser_port = st.text_input("Neo4j Browser Port (HTTP):", value=profile_data.get("browser_port", "7474"), key=f"edit_browser_port_kg_{st.session_state.editing_profile_name}")

                st.markdown("**KG Schema Description File**")
                current_desc_file = profile_data.get("description_file_name")
                if current_desc_file:
                    st.info(f"Current file: `{current_desc_file}`")
                    if st.button("Remove Description File", key=f"remove_desc_kg_{st.session_state.editing_profile_name}"):
                        result = update_kg_profile(
                            profile_name=st.session_state.editing_profile_name,
                            description_file=b""
                        )
                        if result:
                            st.success("Description file removed successfully.")
                            st.rerun()
                        else:
                            st.error("Failed to remove description file. Please check the API logs.")

                edit_desc_file = st.file_uploader("Upload New/Replace KG Description JSON", type=['json'], key=f"edit_desc_upload_kg_{st.session_state.editing_profile_name}")

                st.markdown("**Few-shot examples (Advanced)**")
                _modes = ["hardcoded", "template", "llm"]
                _current_mode = profile_data.get("examples_mode", "hardcoded")
                edit_examples_mode = st.selectbox(
                    "Few-shot examples mode:",
                    options=_modes,
                    index=_modes.index(_current_mode) if _current_mode in _modes else 0,
                    key=f"edit_examples_mode_kg_{st.session_state.editing_profile_name}",
                )
                _current_ip = profile_data.get("identifying_properties") or {}
                edit_identifying_raw = st.text_area(
                    "Identifying properties (JSON mapping label → property):",
                    value=json.dumps(_current_ip, indent=2) if _current_ip else "",
                    placeholder='{"Compound": "name", "Gene": "geneSymbol"}',
                    key=f"edit_identifying_kg_{st.session_state.editing_profile_name}",
                ).strip()

                if st.button("Rebuild dynamic examples now", key=f"rebuild_ex_kg_{st.session_state.editing_profile_name}"):
                    result = rebuild_kg_profile_examples(st.session_state.editing_profile_name)
                    if result:
                        st.success(f"Rebuild triggered: {result}")
                    else:
                        st.error("Failed to rebuild examples. Check the API logs.")

                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Save Changes", key=f"save_edit_kg_{st.session_state.editing_profile_name}"):
                        description_file_content = None
                        if edit_desc_file:
                            description_file_content = edit_desc_file.getvalue()

                        result = update_kg_profile(
                            profile_name=st.session_state.editing_profile_name,
                            host=edit_host,
                            port=edit_port,
                            username=edit_user,
                            password=edit_password,
                            db_name=edit_db_name,
                            browser_port=edit_browser_port,
                            description_file=description_file_content,
                            examples_mode=edit_examples_mode,
                            identifying_properties=edit_identifying_raw if edit_identifying_raw else "",
                        )

                        if result:
                            st.success(f"Profile '{st.session_state.editing_profile_name}' updated.")
                            if st.session_state.selected_kg_profile_name == st.session_state.editing_profile_name:
                                st.rerun()
                        else:
                            st.error("Failed to update profile. Please check the API logs.")

                with col2:
                    if st.button("Delete Profile", type="secondary", key=f"delete_kg_{st.session_state.editing_profile_name}"):
                        if delete_kg_profile(st.session_state.editing_profile_name):
                            st.warning(f"Profile '{st.session_state.editing_profile_name}' deleted.")
                            if st.session_state.selected_kg_profile_name == st.session_state.editing_profile_name:
                                st.session_state.selected_kg_profile_name = None
                            st.session_state.editing_profile_name = None
                            st.rerun()
                        else:
                            st.error("Failed to delete profile. Please check the API logs.")

                with col3:
                    if st.session_state.editing_profile_name and st.session_state.selected_kg_profile_name != st.session_state.editing_profile_name:
                         if st.button(f"Set '{st.session_state.editing_profile_name}' as Active", key=f"activate_kg_{st.session_state.editing_profile_name}"):
                            st.session_state.selected_kg_profile_name = st.session_state.editing_profile_name
                            st.success(f"Profile '{st.session_state.selected_kg_profile_name}' is now active.")
                            st.rerun()
                    elif st.session_state.editing_profile_name and st.session_state.selected_kg_profile_name == st.session_state.editing_profile_name:
                        st.success(f"Profile '{st.session_state.editing_profile_name}' is currently active.")

def render_general_settings_tab():
    st.subheader("General Application Settings")

    st.markdown("##### Conversation Management")
    st.warning("This action is irreversible.")
    if st.button("❌ Delete All Conversations", key="delete_all_threads_settings_button_general", type="primary"):
        if delete_all_threads():
            st.session_state.current_thread_id = None
            st.session_state.data = {}
            st.session_state.just_created_thread = False
            st.session_state.just_posted_message = False
            st.session_state.last_user_message = ""
            st.session_state.title = ""
            if "last_callback_history" in st.session_state:
                del st.session_state.last_callback_history
            st.success("All conversations have been deleted.")
            st.rerun()
        else:
            st.error("Failed to delete conversations. Check API logs.")

def _sync_api_key(widget_key: str, storage_key: str):
    """Copy a widget's value to its persistent session-state key."""
    st.session_state[storage_key] = st.session_state.get(widget_key, "")

def render_api_keys_tab():
    st.subheader("🔑 API Key Management")
    st.info("API keys are stored in the current session only. They will be lost when you close the browser tab.")
    st.caption("Keys entered here override any environment variables configured on the server.")

    ############################
    # OPENAI
    ############################
    st.markdown("##### OpenAI")
    st.text_input(
        "OpenAI API Key:",
        value=st.session_state.get("openai_api_key_stored", ""),
        type="password",
        key="openai_api_key_widget",
        placeholder="sk-...",
        help="Required for OpenAI models (GPT-4, GPT-4o, etc.)",
        on_change=_sync_api_key,
        args=("openai_api_key_widget", "openai_api_key_stored")
    )
    # Sync on first render too
    if st.session_state.get("openai_api_key_widget"):
        st.session_state["openai_api_key_stored"] = st.session_state["openai_api_key_widget"]

    st.markdown("---")

    ############################
    # ANTHROPIC
    ############################
    st.markdown("##### Anthropic")
    st.text_input(
        "Anthropic API Key:",
        value=st.session_state.get("anthropic_api_key_stored", ""),
        type="password",
        key="anthropic_api_key_widget",
        placeholder="sk-ant-...",
        help="Required for Anthropic models",
        on_change=_sync_api_key,
        args=("anthropic_api_key_widget", "anthropic_api_key_stored")
    )
    if st.session_state.get("anthropic_api_key_widget"):
        st.session_state["anthropic_api_key_stored"] = st.session_state["anthropic_api_key_widget"]

    st.markdown("---")

    ############################
    # GOOGLE (GEMINI)
    ############################
    st.markdown("##### Google (Gemini)")
    st.text_input(
        "Google API Key:",
        value=st.session_state.get("google_api_key_stored", ""),
        type="password",
        key="google_api_key_widget",
        placeholder="AIza...",
        help="Required for Google (Gemini) models",
        on_change=_sync_api_key,
        args=("google_api_key_widget", "google_api_key_stored")
    )
    if st.session_state.get("google_api_key_widget"):
        st.session_state["google_api_key_stored"] = st.session_state["google_api_key_widget"]

    st.markdown("---")

    ############################
    # AZURE OPENAI
    ############################
    st.markdown("##### Azure OpenAI")
    st.text_input(
        "Azure API Key:",
        value=st.session_state.get("azure_api_key_stored", ""),
        type="password",
        key="azure_api_key_widget",
        placeholder="...",
        help="Required for Azure OpenAI deployments",
        on_change=_sync_api_key,
        args=("azure_api_key_widget", "azure_api_key_stored")
    )
    if st.session_state.get("azure_api_key_widget"):
        st.session_state["azure_api_key_stored"] = st.session_state["azure_api_key_widget"]

    st.text_input(
        "Azure Endpoint:",
        value=st.session_state.get("azure_endpoint_stored", ""),
        key="azure_endpoint_widget",
        placeholder="https://your-resource.openai.azure.com",
        help="Your Azure OpenAI resource endpoint",
        on_change=_sync_api_key,
        args=("azure_endpoint_widget", "azure_endpoint_stored")
    )
    if st.session_state.get("azure_endpoint_widget"):
        st.session_state["azure_endpoint_stored"] = st.session_state["azure_endpoint_widget"]
