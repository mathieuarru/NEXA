"""Direct mode: the decompose -> generate -> correct agent.

:class:`NexaAgent` is a LangGraph agent whose single tool runs the KG retrieval
pipeline (:class:`~nexa.knowledge_graph.graph_query.GraphQueryAgent`).
:func:`get_llm` builds a chat model for OpenAI, Anthropic, Azure OpenAI, Google
or Ollama.
"""
from typing import Any, Optional, Type, TypedDict, Annotated, List, Generator, Dict
from pydantic import BaseModel, Field, PrivateAttr
from langchain.tools import BaseTool
from nexa.knowledge_graph.graph_query import GraphQueryAgent
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, trim_messages
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from functools import partial
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
import uuid
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.memory import MemorySaver
from psycopg_pool import ConnectionPool
from nexa.models.enums import Status
from nexa.models.prompts import (
    get_agent_prompt_with_tools,
    get_agent_prompt_without_tools,
    get_agent_prompt_for_tool_failure,
    get_agent_prompt_for_no_data_found
)
import json

from langchain_core.callbacks.manager import (
    CallbackManagerForToolRun,
)

from nexa.api.profile_manager import (
    get_profile_with_password,
    load_descriptions_from_profile,
    load_dynamic_examples_for_profile,
    load_identifying_properties_for_profile,
)
from nexa.utils.logger import get_logger
from nexa.config import params
from nexa.models.structured_answers import format_entity_answer
from nexa.knowledge_graph.neo4j_handler import Neo4jHandler
from nexa.utils.message_utils import normalize_message_content

logger = get_logger(__name__)

def get_llm(
    provider: str,
    name: str,
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    azure_api_key: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
) -> BaseChatModel:
    """Create a chat model for ``provider``.

    API keys passed in (e.g. from request headers) take precedence over environment variables.
    """
    logger.info(f"Requesting LLM: Provider={provider}, Name={name}")
    try:
        if provider == "ollama":
            # Ollama defaults to num_ctx=4096 and silently truncates longer
            # prompts, so allocate the window the token budget assumes.
            from nexa.utils.token_utils import get_context_window
            llm = ChatOllama(
                model=name,
                temperature=0,
                base_url=params.OLLAMA_BASE_URL,
                num_ctx=get_context_window(name),
                # Stops repetition loops; see NEXA_LLM_MAX_OUTPUT_TOKENS.
                num_predict=params.NEXA_LLM_MAX_OUTPUT_TOKENS,
            )
        elif provider == "openai":
            api_key = openai_api_key or params.OPENAI_API_KEY
            if not api_key:
                logger.error("OPENAI_API_KEY not found (neither in headers nor environment variables).")
                raise ValueError("OPENAI_API_KEY not found. Please provide it via the UI or environment variables.")

            kwargs = {"model": name, "api_key": api_key, "temperature": 0.0}

            # OpenAI reasoning models (o-series, gpt-5) only accept temperature 1.0.
            reasoning_model_prefixes = ("o1", "o3", "o4", "gpt-5")
            if name.startswith(reasoning_model_prefixes):
                kwargs["temperature"] = 1.0

            logger.info(f"Instantiating ChatOpenAI with parameters: model={name}")
            llm = ChatOpenAI(**kwargs)
        elif provider in {"anthropic", "claude"}:
            api_key = anthropic_api_key or params.ANTHROPIC_API_KEY
            if not api_key:
                logger.error("ANTHROPIC_API_KEY not found (neither in headers nor environment variables).")
                raise ValueError("ANTHROPIC_API_KEY not found. Please provide it via the UI or environment variables.")

            kwargs = {"model": name, "anthropic_api_key": api_key, "temperature": 0.0}
            logger.info(f"Instantiating ChatAnthropic with parameters: model={name}")
            llm = ChatAnthropic(**kwargs)
        elif provider in {"azure", "azure-openai"}:
            api_key = azure_api_key or params.AZURE_API_KEY
            endpoint = azure_endpoint or params.AZURE_OPENAI_ENDPOINT
            if not api_key:
                logger.error("AZURE_API_KEY not found (neither in headers nor environment variables).")
                raise ValueError("AZURE_API_KEY not found. Please provide it via the UI or environment variables.")
            if not endpoint:
                logger.error("AZURE_OPENAI_ENDPOINT not configured.")
                raise ValueError("AZURE_OPENAI_ENDPOINT not found. Please provide it via the UI or environment variables.")

            deployment_name = name or params.AZURE_OPENAI_CHAT_DEPLOYMENT
            if not deployment_name:
                logger.error("No deployment/model provided for the Azure OpenAI provider.")
                raise ValueError("Azure OpenAI provider requires a deployment/model name.")

            kwargs = {
                "azure_deployment": deployment_name,
                "azure_endpoint": endpoint.rstrip("/"),
                "api_key": api_key,
                "api_version": params.AZURE_OPENAI_API_VERSION,
                "temperature": 0.0
            }
            logger.info(f"Instantiating AzureChatOpenAI with deployment={deployment_name}")
            llm = AzureChatOpenAI(**kwargs)
        elif provider in {"gemini", "google"}:
            api_key = google_api_key or params.GOOGLE_API_KEY
            if not api_key:
                logger.error("GOOGLE_API_KEY not found (neither in headers nor environment variables).")
                raise ValueError("GOOGLE_API_KEY not found. Please provide it via the UI or environment variables.")

            kwargs = {"model": name, "google_api_key": api_key, "temperature": 0.0}
            logger.info(f"Instantiating ChatGoogleGenerativeAI with model={name}")
            llm = ChatGoogleGenerativeAI(**kwargs)
        else:
            logger.warning(f"Unsupported LLM provider requested: {provider}")
            raise ValueError(f"Unsupported LLM provider: {provider}")

        return llm
    except Exception as e:
        logger.error(f"Error creating LLM instance {provider}/{name}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to create LLM instance {provider}/{name}: {str(e)}")


############################
# AGENT CLASS
############################

class QueryResult(TypedDict):
    """The structured output format for the simple query method."""
    answer: str
    cypher_query: Optional[str]
    cypher_first_attempt: Optional[str]
    n_correction_iterations: Optional[int]
    n_neo4j_error_corrections: Optional[int]
    fallback_triggered: Optional[bool]
    thinking: List[str]
    data: Optional[any]
    kg_results: Optional[any]
    error: Optional[str]

class AgentState(TypedDict):
    """LangGraph state for the Direct-mode conversation graph: message history plus the
    optional KG profile, custom system prompt, and structured-output schema/name."""
    messages: Annotated[List, add_messages]
    title: Optional[str]
    kg_profile_name: Optional[str]
    custom_system_prompt: Optional[str]
    structured_output_schema_json: Optional[str]
    structured_output_name: Optional[str]

class NexaAgent:
    """Direct-mode agent: an LLM that answers with the KG retrieval pipeline as its tool."""
    _llm: BaseChatModel = PrivateAttr()

    def __init__(self, llm: BaseChatModel, db_uri: Optional[str] = None, rephrasing: bool = False,
                 neo4j_host: str = None,
                 neo4j_port: str = None,
                 neo4j_username: str = None,
                 neo4j_password: str = None,
                 neo4j_db_name: str = "neo4j",
                 nodes_descriptions: Optional[dict] = None,
                 relationships_descriptions: Optional[dict] = None,
                 kg_profile_name_for_thread: Optional[str] = None,
                 connection_pool: Optional[ConnectionPool] = None,
                 checkpointer = None,
                 dynamic_examples: Optional[dict] = None,
                 identifying_properties: Optional[dict] = None,
                 generation_mode: str = "one_shot_fallback"
                 ):
        self._llm = llm
        self.rephrasing = rephrasing
        self.db_uri = db_uri

        if checkpointer:
            self.checkpointer = checkpointer
            logger.info("Using provided checkpointer.")
        elif db_uri:
            pool = connection_pool if connection_pool is not None else self.setup_postgres_pool(db_uri)
            self.checkpointer = PostgresSaver(pool)
            logger.info("Initialized PostgresSaver for stateful conversation.")
        else:
            self.checkpointer = MemorySaver()
            logger.info("Initialized MemorySaver for stateless operation.")

        self._neo4j_host = neo4j_host
        self._neo4j_port = neo4j_port
        self._neo4j_username = neo4j_username
        self._neo4j_password = neo4j_password
        self._neo4j_db_name = neo4j_db_name
        self._nodes_descriptions = nodes_descriptions
        self._relationships_descriptions = relationships_descriptions
        self._kg_profile_name_for_thread = kg_profile_name_for_thread
        self._dynamic_examples = dynamic_examples or {}
        self._identifying_properties = identifying_properties or {}
        self._generation_mode = generation_mode or "one_shot_fallback"

    @classmethod
    def from_config(
        cls,
        model_provider: str,
        model_name: str,
        db_uri: Optional[str] = None,
        kg_profile_name: Optional[str] = None,
        rephrasing: bool = False,
        connection_pool: Optional[ConnectionPool] = None,
        checkpointer = None,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        examples_mode_override: Optional[str] = None,
        generation_mode: str = "one_shot_fallback",
    ):
        """Create an agent from a KG profile name and model settings.

        API keys passed in take precedence over environment variables.
        """
        agent_init_kwargs = {
            "db_uri": db_uri,
            "rephrasing": rephrasing,
            "kg_profile_name_for_thread": kg_profile_name,
            "connection_pool": connection_pool,
            "checkpointer": checkpointer,
            "generation_mode": generation_mode or "one_shot_fallback",
        }

        nodes_descriptions_content: Optional[dict] = None
        relationships_descriptions_content: Optional[dict] = None
        dynamic_examples_content: Optional[dict] = None
        identifying_properties_content: Optional[dict] = None

        if kg_profile_name:
            profile_data = get_profile_with_password(kg_profile_name)
            if not profile_data:
                raise ValueError(f"KG Profile '{kg_profile_name}' not found.")

            agent_init_kwargs["neo4j_host"] = profile_data.get("host")
            agent_init_kwargs["neo4j_port"] = profile_data.get("port")
            agent_init_kwargs["neo4j_username"] = profile_data.get("username")
            agent_init_kwargs["neo4j_password"] = profile_data.get("password")
            agent_init_kwargs["neo4j_db_name"] = profile_data.get("db_name")

            nodes_descriptions_content, relationships_descriptions_content = load_descriptions_from_profile(profile_data, kg_profile_name)

            profile_mode = profile_data.get("examples_mode", "hardcoded")
            # Session-level override (from sidebar) wins over the profile default.
            examples_mode = examples_mode_override if examples_mode_override in ("hardcoded", "template", "llm") else profile_mode
            dynamic_examples_content = load_dynamic_examples_for_profile(kg_profile_name, examples_mode)
            identifying_properties_content = load_identifying_properties_for_profile(kg_profile_name)
            if dynamic_examples_content:
                src = "override" if examples_mode_override and examples_mode == examples_mode_override else "profile"
                logger.info(f"Loaded dynamic examples for profile '{kg_profile_name}' (mode={examples_mode}, source={src}).")
            elif examples_mode != "hardcoded":
                logger.warning(
                    f"Requested examples_mode='{examples_mode}' for profile '{kg_profile_name}' "
                    f"has no cache entry; falling back to the hardcoded defaults."
                )
        else:
            logger.info("No KG profile specified. NexaAgent will use default/environment variable Neo4j connection settings.")

        agent_init_kwargs["nodes_descriptions"] = nodes_descriptions_content
        agent_init_kwargs["relationships_descriptions"] = relationships_descriptions_content
        agent_init_kwargs["dynamic_examples"] = dynamic_examples_content
        agent_init_kwargs["identifying_properties"] = identifying_properties_content

        llm_instance = get_llm(
            provider=model_provider,
            name=model_name,
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            google_api_key=google_api_key,
            azure_api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
        )
        agent_init_kwargs["llm"] = llm_instance

        return cls(**agent_init_kwargs)

    @classmethod
    def from_credentials(
        cls,
        model_provider: str,
        model_name: str,
        neo4j_host: str,
        neo4j_username: str,
        neo4j_password: str,
        neo4j_port: str = "7687",
        neo4j_db_name: str = "neo4j",
        db_uri: Optional[str] = None,
        rephrasing: bool = False,
        nodes_descriptions: Optional[dict] = None,
        relationships_descriptions: Optional[dict] = None,
        connection_pool: Optional[ConnectionPool] = None,
        checkpointer=None
    ):
        """Create an agent from explicit Neo4j credentials, without a KG profile."""
        llm_instance = get_llm(provider=model_provider, name=model_name)

        return cls(
            llm=llm_instance,
            db_uri=db_uri,
            rephrasing=rephrasing,
            neo4j_host=neo4j_host,
            neo4j_port=neo4j_port,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password,
            neo4j_db_name=neo4j_db_name,
            nodes_descriptions=nodes_descriptions,
            relationships_descriptions=relationships_descriptions,
            kg_profile_name_for_thread=None,
            connection_pool=connection_pool,
            checkpointer=checkpointer
        )


############################
# POSTGRES CONNECTION
############################

    @staticmethod
    def setup_postgres_pool(db_uri: str) -> ConnectionPool:
        """Initialize a PostgreSQL connection pool."""
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
        }
        try:
            pool = ConnectionPool(conninfo=db_uri, max_size=20, kwargs=connection_kwargs)
            logger.info("PostgreSQL pool initialized.")
            return pool
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL pool: {e}")
            raise RuntimeError(f"Failed to initialize PostgreSQL pool: {e}")

############################
# CONVERSATION TITLE GENERATION
############################

    def generate_conversation_title(self, first_message: str) -> str:
        """Generate a short conversation title from the first user message."""
        class GeneratedTitle(BaseModel):
            """
            Represents a generated conversation title.
            """
            title: str = Field(description=("Short (max 10 words), descriptive title in English."))

        system_prompt = """You are a title generator. Create a short, descriptive title (max 10 words) in English for a conversation based on the user's first message below."""

        try:
            structured_llm = self._llm.with_structured_output(GeneratedTitle)
            response = structured_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=first_message)
            ])
            return response.title
        except Exception as e:
            logger.error(f"Error generating title: {e}")
            return f"Conversation ({first_message[:20]}...)"

############################
# AGENT GRAPH BUILDING
############################

    def build_agent_graph(self, kg_mode: str, output_schema: Optional[Type[BaseModel]] = None, stop_check=None) -> StateGraph:
        """Build and compile the agent graph for ``kg_mode``.

        ``stop_check`` is passed to the KG tool, which calls it between pipeline
        steps so a user stop takes effect mid-pipeline.
        """
        builder = StateGraph(AgentState)

        tools = []
        if kg_mode != "Off":
            logger.info(f"KG Mode is '{kg_mode}', enabling RetrieveDataFromKG tool.")
            kg_agent_instance = GraphQueryAgent(
                llm=self._llm,
                rephrasing=self.rephrasing,
                neo4j_host=self._neo4j_host,
                neo4j_port=self._neo4j_port,
                neo4j_username=self._neo4j_username,
                neo4j_password=self._neo4j_password,
                neo4j_db_name=self._neo4j_db_name,
                nodes_descriptions=self._nodes_descriptions,
                relationships_descriptions=self._relationships_descriptions,
                dynamic_examples=self._dynamic_examples,
                identifying_properties=self._identifying_properties,
                generation_mode=self._generation_mode,
            )
            kg_tool = self.RetrieveDataFromKG(kg_agent_instance)
            if stop_check is not None:
                kg_tool.set_stop_check(stop_check)
            tools = [kg_tool]
        else:
             logger.info("KG Mode is 'Off', skipping KG tool.")

        agent_node_func = partial(self.agent_node, tools=tools, kg_mode=kg_mode, output_schema=output_schema)
        builder.add_node("agent", agent_node_func)
        builder.add_node("tools", ToolNode(tools))

        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", tools_condition)
        builder.add_edge("tools", "agent")

        return builder.compile(checkpointer=self.checkpointer)

############################
# AGENT NODE
############################

    def agent_node(self, state: AgentState, tools: list, kg_mode: str, output_schema: Optional[Type[BaseModel]] = None):
        """Run one LLM turn.

        Formats KG tool results to the token budget, picks the system prompt
        (default, custom, tool failure or empty result) and calls the LLM with
        or without tools according to ``kg_mode``.
        """
        original_messages_history = state.get("messages", [])
        custom_system_prompt = state.get("custom_system_prompt")

        # The KG result block and the trimmed history share one budget sized to
        # the model's context window.
        from nexa.utils.token_utils import (
            compute_kg_budget, get_context_window, get_model_name,
        )
        model_name = get_model_name(self._llm)
        context_window = get_context_window(model_name)
        kg_budget = compute_kg_budget(context_window)
        logger.debug(
            f"[token_budget] model={model_name!r} window={context_window} kg_budget={kg_budget}"
        )

        messages_history = []
        for msg in original_messages_history:
            if isinstance(msg, ToolMessage):
                try:
                    tool_content = json.loads(msg.content)
                    if isinstance(tool_content, dict):
                        formatted_content = self._format_kg_results_compact(
                            tool_content, max_tokens=kg_budget, model_name=model_name,
                        )
                        formatted_msg = ToolMessage(
                            content=formatted_content,
                            tool_call_id=msg.tool_call_id,
                            name=msg.name
                        )
                        messages_history.append(formatted_msg)
                        logger.info(f"Formatted KG result for tool_call_id: {msg.tool_call_id}")
                    else:
                        messages_history.append(msg)
                except (json.JSONDecodeError, TypeError):
                    messages_history.append(msg)
            else:
                messages_history.append(msg)

        # Drop the oldest messages when the history exceeds the budget. The kept
        # history may start on a human or a tool message.
        prev_count = len(messages_history)
        try:
            messages_history = trim_messages(
                messages_history,
                max_tokens=kg_budget,
                token_counter="approximate",
                strategy="last",
                start_on=("human", "tool"),
                include_system=False,
            )
        except Exception as trim_err:
            logger.warning(f"[trim_messages] failed ({trim_err}); falling back to start_on='human'")
            try:
                messages_history = trim_messages(
                    messages_history,
                    max_tokens=kg_budget,
                    token_counter="approximate",
                    strategy="last",
                    start_on="human",
                    include_system=False,
                )
            except Exception as fallback_err:
                logger.warning(f"[trim_messages] fallback also failed ({fallback_err}); using untrimmed history")
        if len(messages_history) < prev_count:
            logger.info(f"[trim_messages] dropped {prev_count - len(messages_history)} old messages to fit budget")

        system_prompt_content = None
        last_message = original_messages_history[-1] if original_messages_history else None
        custom_prompt_for_tool_issue_used = False

        if isinstance(last_message, ToolMessage) and tools:
            try:
                tool_content = json.loads(last_message.content)
                if isinstance(tool_content, dict):
                    if "error" in tool_content:
                        if custom_system_prompt:
                            system_prompt_content = (
                                custom_system_prompt
                                + "\n\n[Note: The KG retrieval pipeline failed before returning evidence. "
                                "Do not invent KG results; answer according to the output format above based on the available context.]"
                            )
                            logger.info("Tool returned an error. Augmenting custom system prompt with failure note.")
                        else:
                            system_prompt_content = get_agent_prompt_for_tool_failure()
                            logger.info("Tool returned an error. Using default prompt for tool failure.")
                        custom_prompt_for_tool_issue_used = True
                    elif tool_content.get("message") == "The query executed successfully but found no matching data.":
                        n_iter = tool_content.get("n_correction_iterations") or 0
                        attempts_note = (
                            f" The system attempted {n_iter} correction iteration(s) "
                            f"(synonym search, alternative literals) before concluding."
                            if n_iter > 0 else ""
                        )
                        if custom_system_prompt:
                            system_prompt_content = (
                                custom_system_prompt
                                + f"\n\n[Note: The KG query executed but returned no matching rows.{attempts_note} "
                                "This means either the queried information is not present in the knowledge graph, "
                                "or the pipeline was unable to formulate a Cypher query that retrieves it. "
                                "If the user question asks whether something is the case (yes/no), treat the empty result as the absence of supporting evidence: "
                                "an empty result confirms a negation, and disconfirms a positive claim. "
                                "If the user question is open-ended, report that no entities matched.]"
                            )
                            logger.info(f"Tool found no data after {n_iter} correction iterations. Augmenting custom system prompt with empty-result note.")
                        else:
                            system_prompt_content = get_agent_prompt_for_no_data_found()
                            logger.info("Tool found no data. Using default prompt for no data found.")
                        custom_prompt_for_tool_issue_used = True
            except json.JSONDecodeError:
                logger.warning(f"Could not parse last ToolMessage content for error checking: {last_message.content}")
            except Exception as e:
                logger.error(f"Error while processing last ToolMessage for custom prompt: {e}")

        if not custom_prompt_for_tool_issue_used:
            if custom_system_prompt:
                system_prompt_content = custom_system_prompt
                logger.debug("Agent node executing with custom system prompt.")
            elif not tools:
                system_prompt_content = get_agent_prompt_without_tools()
                logger.debug("Agent node executing without tools (KG Mode: Off). Using AGENT_SYSTEM_PROMPT_WITHOUT_TOOLS.")
            else:
                system_prompt_content = get_agent_prompt_with_tools()
                logger.debug(f"Agent node executing with tools (KG Mode: {kg_mode}). Using AGENT_SYSTEM_PROMPT_WITH_TOOLS.")

        system_message = SystemMessage(content=system_prompt_content)

        if output_schema and isinstance(last_message, ToolMessage):
            logger.info(f"Using structured output with schema '{output_schema.__name__}' to process tool results.")
            structured_llm = self._llm.with_structured_output(output_schema)
            try:
                raw_response = structured_llm.invoke([system_message] + messages_history)
                response = AIMessage(content=raw_response.model_dump_json())
            except Exception as struct_err:
                # A long entity list can exceed the output-token limit; then the
                # answer is built from the KG result. Other errors are re-raised.
                response = self._handle_structured_output_length_overflow(
                    err=struct_err,
                    last_tool_message=last_message,
                    output_schema=output_schema,
                )
                if response is None:
                    raise
        else:
            if kg_mode == "Off":
                logger.info("KG Mode is 'Off', using direct invocation.")
                response = self._llm.invoke([system_message] + messages_history)
            elif kg_mode == "Auto":
                logger.info("KG Mode is 'Auto', using tools if available.")
                if custom_prompt_for_tool_issue_used:
                    logger.info("Custom prompt for tool issue was used. Invoking LLM for response to user without tools.")
                    response = self._llm.invoke([system_message] + messages_history)
                else:
                    llm_with_tools = self._llm.bind_tools(tools)
                    response = llm_with_tools.invoke([system_message] + messages_history)
            elif kg_mode == "Force":
                tool_name_to_force = tools[0].name if tools else None
                if custom_prompt_for_tool_issue_used:
                    logger.info("Custom prompt for tool issue was used. Invoking LLM for response to user, not forcing tool.")
                    response = self._llm.invoke([system_message] + messages_history)
                elif tool_name_to_force and isinstance(last_message, ToolMessage) and last_message.tool_call_id:
                    logger.info(f"KG Mode is 'Force', received ToolMessage result. Invoking LLM for final response.")
                    llm_with_tools = self._llm.bind_tools(tools)
                    response = llm_with_tools.invoke([system_message] + messages_history)
                elif tool_name_to_force:
                    logger.info(f"KG Mode is 'Force'. Forcing tool call: {tool_name_to_force}")
                    response = self._llm.bind_tools(tools, tool_choice=tool_name_to_force).invoke([system_message] + messages_history)
                else:
                    logger.warning("KG Mode is 'Force' but no tool could be identified. Falling back to direct invocation.")
                    response = self._llm.invoke([system_message] + messages_history)
            else:
                logger.warning(f"Unexpected kg_mode '{kg_mode}' received in agent_node. Defaulting to 'Auto' behavior.")
                if custom_prompt_for_tool_issue_used:
                    logger.info("Custom prompt for tool issue was used. Invoking LLM for response to user without tools.")
                    response = self._llm.invoke([system_message] + messages_history)
                else:
                    llm_with_tools = self._llm.bind_tools(tools)
                    response = llm_with_tools.invoke([system_message] + messages_history)


        if hasattr(response, 'tool_calls') and response.tool_calls:
            original_human_query_content = None
            for msg in reversed(messages_history):
                if isinstance(msg, HumanMessage):
                    original_human_query_content = msg.content
                    break

            modified_tool_calls = []
            for tc in response.tool_calls:
                tool_call_dict = tc if isinstance(tc, dict) else tc.dict()

                if tool_call_dict.get('name') == 'retrieve_data_from_kg':
                    if not isinstance(tool_call_dict.get('args'), dict):
                        tool_call_dict['args'] = {}

                    llm_query = tool_call_dict['args'].get('user_query')
                    llm_query_valid = isinstance(llm_query, str) and llm_query.strip()

                    if params.NEXA_FORCE_INJECT_USER_QUERY and original_human_query_content is not None:
                        logger.info(
                            f"FORCE override: LLM generated user_query={llm_query!r} -> "
                            f"replaced with original {original_human_query_content!r}"
                        )
                        tool_call_dict['args']['user_query'] = original_human_query_content
                    elif not llm_query_valid:
                        if original_human_query_content is not None:
                            logger.warning(
                                f"LLM did not provide a valid user_query (got {llm_query!r}); "
                                f"falling back to injection with {original_human_query_content!r}"
                            )
                            tool_call_dict['args']['user_query'] = original_human_query_content
                        else:
                            logger.warning(
                                "LLM did not provide user_query and no HumanMessage found to fall back on; "
                                "tool may fail."
                            )
                    else:
                        logger.info(f"Using LLM-generated user_query: {llm_query!r}")

                modified_tool_calls.append(tool_call_dict)
            response.tool_calls = modified_tool_calls

        logger.info(f"Agent response (tool_calls potentially injected): {response}")
        return {'messages': [response]}

############################
# TOOL KG RETRIEVAL
############################

    class RetrieveDataFromKG(BaseTool):
        """Tool to retrieve data from the Knowledge Graph."""
        class RetrieveDataFromKGInput(BaseModel):
            user_query: str = Field(description="The user's natural-language question to run against the Knowledge Graph. Pass the user's original question as-is unless reformulation is clearly needed.")

        name: str = "retrieve_data_from_kg"
        description: str = "Retrieves data from the Knowledge Graph to answer the user's question. Provide the user's original question in the user_query argument."
        args_schema: Type[BaseModel] = RetrieveDataFromKGInput
        _kg_agent: GraphQueryAgent = PrivateAttr()
        # Called by run_entire_graph between pipeline steps to honor user stops.
        _stop_check: Optional[Any] = PrivateAttr(default=None)

        def __init__(self, kg_agent: GraphQueryAgent):
            super().__init__()
            self._kg_agent = kg_agent

        def set_stop_check(self, stop_check) -> None:
            """Set the stop-check callable the KG pipeline calls between steps."""
            self._stop_check = stop_check

        def _run(self, user_query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> dict:
            """Run the KG pipeline on ``user_query`` and return its parsed result."""
            logger.info(f"Tool: Running KG query for: '{user_query}'")
            try:
                final_state = self._kg_agent.run_entire_graph(
                    user_query, stop_check=self._stop_check
                )

                logger.info("Tool: run_entire_graph execution complete.")
                logger.debug(
                    "Tool: run_entire_graph state keys: %s",
                    list(final_state.keys()) if isinstance(final_state, dict) else type(final_state).__name__,
                )

                return self.parse_kg_state(final_state)

            except Exception as e:
                # A user stop must reach stream_generator, not become a tool error.
                from nexa.api.stop_registry import UserStoppedException
                if isinstance(e, UserStoppedException):
                    raise
                logger.error(f"Tool error during run_entire_graph execution: {e}", exc_info=True)
                return {"error": f"An unexpected error occurred running the KG pipeline: {str(e)}"}

        def parse_kg_state(self, state: dict) -> dict:
            """Build the tool result (data, Cypher query or error, plus telemetry) from the final pipeline state."""
            final_status = state.get("status")
            error_msg = state.get("error")
            data = state.get("data")
            cypher_queries = state.get("cypher_queries") or []
            attempts_info = state.get("attempts_info") or {}
            n_correction_iterations = (attempts_info.get("global_") or {}).get("current")
            # Neo4j-error repair rounds, counted separately from property repairs.
            n_neo4j_error_corrections = (attempts_info.get("neo4j_error") or {}).get("current")
            first_query = cypher_queries[0] if cypher_queries else None

            telemetry = {
                "cypher_first_attempt": first_query,
                "n_correction_iterations": n_correction_iterations,
                "n_neo4j_error_corrections": n_neo4j_error_corrections,
                # one_shot_fallback only; None in the other generation modes.
                "fallback_triggered": state.get("fallback_triggered"),
            }

            if error_msg:
                logger.error(f"Tool: KG pipeline finished with error: {error_msg}")
                return {"error": error_msg, **telemetry}
            elif final_status == Status.SUCCESS or final_status == Status.DONE or data:
                last_query = cypher_queries[-1] if cypher_queries else None
                if data:
                    logger.info("Tool: KG pipeline finished successfully with data.")
                    result = {"data": data, **telemetry}
                    if last_query:
                        result["cypher_query"] = last_query
                        logger.debug(f"Tool: Extracted last Cypher query: {last_query}")
                    return result
                else:
                    logger.warning(f"Tool: KG pipeline finished successfully (Status: {final_status}) but returned no data.")
                    result = {"message": "The query executed successfully but found no matching data.", **telemetry}
                    if last_query:
                        result["cypher_query"] = last_query
                        logger.debug(f"Tool: Extracted last Cypher query (no data): {last_query}")
                    return result
            else:
                unknown_error = "Unknown state after KG pipeline execution."
                logger.error(f"Tool: {unknown_error} Final status: {final_status}")
                return {"error": unknown_error, **telemetry}


    def _format_kg_results_compact(
        self, kg_data: dict, display_name_keys: list = None, max_tokens: int | None = None,
        model_name: str = "",
    ) -> str:
        """Format a KG result for the final LLM call.

        Without ``max_tokens``, returns the full table (level 0). Otherwise falls
        back to names only (level 1), then a count summary (level 2), to fit the
        budget. Level 2 is returned even if it is still over budget.
        """
        if not kg_data or not isinstance(kg_data, dict):
            return "No data received from Knowledge Graph."

        if kg_data.get("error"):
            return f"ERROR: {kg_data.get('error')}"

        data = kg_data.get("data")
        nodes = data.get("nodes", []) if data else []
        edges = data.get("edges", []) if data else []
        cypher_query = kg_data.get("cypher_query")
        message = kg_data.get("message")

        if not nodes:
            parts = []
            if cypher_query:
                parts.append(f"CYPHER: {cypher_query}")
                parts.append("The above Cypher query was executed on the knowledge graph.")
            parts.append(message or "The query executed successfully but found no matching data.")
            return "\n".join(parts)

        formatted = self._kg_level_0_full(kg_data, display_name_keys)

        if max_tokens is None:
            return formatted

        from nexa.utils.token_utils import count_tokens

        tokens_0 = count_tokens(formatted, model=model_name)
        if tokens_0 <= max_tokens:
            return formatted

        formatted_l1 = self._kg_level_1_names_only(kg_data, display_name_keys)
        tokens_1 = count_tokens(formatted_l1, model=model_name)
        if tokens_1 <= max_tokens:
            logger.info(
                f"[kg_compaction] level=1 ({len(nodes)} nodes, {len(edges)} edges): "
                f"level_0={tokens_0} > budget {max_tokens}, level_1={tokens_1}"
            )
            return formatted_l1

        formatted_l2 = self._kg_level_2_summary(kg_data, display_name_keys)
        tokens_2 = count_tokens(formatted_l2, model=model_name)
        if tokens_2 > max_tokens:
            logger.warning(
                f"[kg_compaction] level=2 still over budget ({tokens_2}/{max_tokens}); sending anyway"
            )
        else:
            logger.info(
                f"[kg_compaction] level=2 ({len(nodes)} nodes, {len(edges)} edges): "
                f"level_0={tokens_0} level_1={tokens_1} > budget {max_tokens}, level_2={tokens_2}"
            )
        return formatted_l2

    def _kg_level_0_full(self, kg_data: dict, display_name_keys: list = None) -> str:
        """Render the KG result as full tables with every property (level 0)."""
        data = kg_data.get("data")
        cypher_query = kg_data.get("cypher_query")
        nodes = data.get("nodes", []) if data else []
        edges = data.get("edges", []) if data else []

        result_parts = []
        if cypher_query:
            result_parts.append("The following Cypher query was executed on the knowledge graph and retrieved the information below:")
            result_parts.append(f"CYPHER: {cypher_query}\n")

        edges_by_type = {}
        for edge in edges:
            edge_type = edge.get('label', 'RELATED_TO')
            if edge_type not in edges_by_type:
                edges_by_type[edge_type] = []
            edges_by_type[edge_type].append(edge)

        edge_analysis = {}
        if edges_by_type:
            result_parts.append("# SUMMARY")

            for edge_type, type_edges in edges_by_type.items():
                source_degrees = {}
                target_degrees = {}

                for edge in type_edges:
                    src_id = edge.get('source_node', '')
                    dst_id = edge.get('target_node', '')
                    source_degrees[src_id] = source_degrees.get(src_id, 0) + 1
                    target_degrees[dst_id] = target_degrees.get(dst_id, 0) + 1

                avg_source_degree = sum(source_degrees.values()) / len(source_degrees) if source_degrees else 0
                avg_target_degree = sum(target_degrees.values()) / len(target_degrees) if target_degrees else 0

                degree_ratio = avg_target_degree / avg_source_degree if avg_source_degree > 0 else 1
                use_target_centric = degree_ratio > 2.0

                if use_target_centric:
                    edges_by_target = {}
                    for edge in type_edges:
                        src_id = edge.get('source_node', '')
                        dst_id = edge.get('target_node', '')
                        if dst_id not in edges_by_target:
                            edges_by_target[dst_id] = []
                        edges_by_target[dst_id].append(src_id)

                    edge_analysis[edge_type] = ("target_centric", edges_by_target)

                    source_appearances = {}
                    for tgt_id, src_list in edges_by_target.items():
                        for src_id in src_list:
                            if src_id not in source_appearances:
                                source_appearances[src_id] = []
                            source_appearances[src_id].append(tgt_id)

                    shared_entities = {src_id: tgt_list for src_id, tgt_list in source_appearances.items() if len(tgt_list) > 1}
                else:
                    edges_by_source = {}
                    for edge in type_edges:
                        src_id = edge.get('source_node', '')
                        dst_id = edge.get('target_node', '')
                        if src_id not in edges_by_source:
                            edges_by_source[src_id] = []
                        edges_by_source[src_id].append(dst_id)

                    edge_analysis[edge_type] = ("source_centric", edges_by_source)

                    target_appearances = {}
                    for src_id, dst_list in edges_by_source.items():
                        for dst_id in dst_list:
                            if dst_id not in target_appearances:
                                target_appearances[dst_id] = []
                            target_appearances[dst_id].append(src_id)

                    shared_entities = {dst_id: src_list for dst_id, src_list in target_appearances.items() if len(src_list) > 1}

                total_relations = len(type_edges)
                shared_count = len(shared_entities)
                pivot_type = "target-pivot" if use_target_centric else "source-pivot"

                if shared_entities:
                    shared_list = ', '.join(map(str, sorted(shared_entities.keys())))

                    all_connected_entities = set()
                    for connected_list in shared_entities.values():
                        all_connected_entities.update(connected_list)

                    between_entities = ' and '.join(map(str, sorted(all_connected_entities)))
                    result_parts.append(f"Relations: {total_relations} total | Shared entities between {between_entities} (n={shared_count}): {shared_list} | Format: {pivot_type}")
                else:
                    result_parts.append(f"Relations: {total_relations} total | Shared entities: none | Format: {pivot_type}")
                result_parts.append("")

        nodes_by_type = {}
        for node in nodes:
            node_type = node.get('label', 'Entity')
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append(node)

        for node_type, type_nodes in nodes_by_type.items():
            if not type_nodes:
                continue

            all_props = set()
            for node in type_nodes:
                all_props.update(node.keys())

            # `element_id` is only for traceability; `id` is the join key used by
            # the edge blocks below.
            system_props = {'id', 'element_id', 'label'}
            data_props = sorted(all_props - system_props)

            columns = ['id'] + data_props
            header = f"# NODES.{node_type} ({' | '.join(columns)})"
            result_parts.append(header)

            for node in type_nodes:
                row_values = []
                for col in columns:
                    value = node.get(col, '')

                    if isinstance(value, str):
                        value = value.replace('|', '¦')
                    elif value is None:
                        value = ''
                    row_values.append(str(value))
                result_parts.append('|'.join(row_values))

            result_parts.append("")

        for edge_type, (format_type, adjacency_data) in edge_analysis.items():
            if not adjacency_data:
                continue

            if format_type == "target_centric":
                header = f"# EDGES.{edge_type} (target_id <- [source_id_list])"
                connector = " <- "
                count_label = "sources"
            else:
                header = f"# EDGES.{edge_type} (source_id -> [target_id_list])"
                connector = " -> "
                count_label = "targets"

            result_parts.append(header)

            for pivot_id, connected_ids in adjacency_data.items():
                count = len(connected_ids)
                if count <= 8:
                    ids_list_str = f"[{', '.join(map(str, connected_ids))}]"
                    result_parts.append(f"{pivot_id}{connector}{ids_list_str}          # {count} {count_label}")
                else:
                    lines = []
                    for i in range(0, count, 8):
                        chunk = connected_ids[i:i+8]
                        if i == 0:
                            lines.append(f"{pivot_id}{connector}[{', '.join(map(str, chunk))},")
                        elif i + 8 >= count:
                            lines.append(f"         {', '.join(map(str, chunk))}]   # {count} {count_label}")
                        else:
                            lines.append(f"         {', '.join(map(str, chunk))},")
                    result_parts.extend(lines)

            result_parts.append("")

        return "\n".join(result_parts).strip()

    def _handle_structured_output_length_overflow(
        self,
        err: Exception,
        last_tool_message: ToolMessage,
        output_schema: Type[BaseModel],
    ) -> Optional[AIMessage]:
        """Build the answer from the KG result when structured output hit the output-token limit.

        Only for schemas whose ``answer`` field is an entity list; the shape
        (CSV string or list) comes from :func:`format_entity_answer`. Returns
        None when the error is not a length overflow or the schema does not
        fit, and the caller then re-raises.
        """
        # Length-overflow signatures from OpenAI, Anthropic and Gemini, matched by
        # class name and message so that openai need not be imported.
        err_msg = str(err).lower()
        cls_name = type(err).__name__
        is_length_overflow = (
            cls_name == "LengthFinishReasonError"                      # OpenAI (class)
            or "length limit was reached" in err_msg                   # OpenAI message
            or "finish_reason='length'" in err_msg                     # OpenAI (langchain wrap)
            or "max_tokens" in err_msg                                 # Anthropic stop_reason / Gemini finish_reason
            or "max tokens" in err_msg                                 # generic phrasing
            or "stop_reason='max_tokens'" in err_msg                   # Anthropic explicit
            or "finish_reason='max_tokens'" in err_msg                 # Gemini explicit
            # Truncated Anthropic tool call: the required `answer` field is
            # missing. Checked narrowly so real schema mismatches are not caught.
            or (
                cls_name == "ValidationError"
                and "field required" in err_msg
                and "\nanswer\n" in err_msg            # missing field is `answer`
            )
        )
        if not is_length_overflow:
            return None

        try:
            tool_content = json.loads(last_tool_message.content)
        except (TypeError, json.JSONDecodeError):
            logger.warning("[length-overflow] could not parse tool content for deterministic extraction.")
            return None
        kg_data = tool_content.get("data") or {}
        nodes = kg_data.get("nodes", []) if isinstance(kg_data, dict) else []

        # Name each entity by the profile's identifying property for its label
        # (e.g. Gene -> geneSymbol), falling back to generic display keys.
        identifying = self._identifying_properties or {}
        display_name_keys = ['commonName', 'name', 'geneSymbol', 'symbol', 'title']
        names: List[str] = []
        for n in nodes:
            if not isinstance(n, dict):
                continue
            val = None
            label = n.get("label")
            if label:
                prop = identifying.get(label)
                if prop:
                    key = f"{prop}_property" if prop in Neo4jHandler._RESERVED_NODE_KEYS else prop
                    val = n.get(key)
            if not val:
                for k in display_name_keys:
                    v = n.get(k)
                    if v:
                        val = v
                        break
            if val:
                names.append(str(val))

        # Deduplicate, keeping order.
        seen, deduped = set(), []
        for nm in names:
            if nm not in seen:
                seen.add(nm)
                deduped.append(nm)

        if not deduped:
            logger.warning("[length-overflow] no entity names could be extracted; re-raising.")
            return None

        # None when the schema has no entity-list `answer` field.
        synthetic = format_entity_answer(
            output_schema, deduped,
            explanation=(
                "The LLM output reached the max-output-tokens limit while listing entities; "
                f"the {len(deduped)} entity names were extracted directly from the KG result set."
            ),
        )
        if synthetic is None:
            logger.warning(
                f"[length-overflow] LLM output truncated but schema {output_schema.__name__!r} "
                f"has no entity-list `answer` field; re-raising."
            )
            return None

        logger.warning(
            f"[length-overflow] LLM output truncated on {output_schema.__name__}. "
            f"Falling back to deterministic extraction of {len(deduped)} entity names from the KG result."
        )
        return AIMessage(content=synthetic.model_dump_json())

    def _kg_level_1_names_only(self, kg_data: dict, display_name_keys: list = None) -> str:
        """Render node display names by type and edges as adjacency lists (level 1)."""
        data = kg_data.get("data") or {}
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        cypher_query = kg_data.get("cypher_query")

        if display_name_keys is None:
            display_name_keys = ['commonName', 'name', 'geneSymbol', 'symbol', 'title']

        parts = []
        if cypher_query:
            parts.append("The following Cypher query was executed on the knowledge graph and retrieved the information below:")
            parts.append(f"CYPHER: {cypher_query}\n")

        node_name_map = {}
        nodes_by_type: dict = {}
        for node in nodes:
            node_id = str(node.get('id', ''))
            readable_name = next((node.get(k) for k in display_name_keys if node.get(k)), f"Node_{node_id}")
            node_name_map[node_id] = readable_name
            nodes_by_type.setdefault(node.get('label', 'Entity'), []).append(readable_name)

        for node_type, names in nodes_by_type.items():
            parts.append(f"# NODES.{node_type} ({len(names)} entities): {', '.join(map(str, names))}")
            parts.append("")

        edges_by_type: dict = {}
        for edge in edges:
            edges_by_type.setdefault(edge.get('label', 'RELATED_TO'), []).append(edge)

        for edge_type, type_edges in edges_by_type.items():
            adjacency: dict = {}
            for edge in type_edges:
                src = node_name_map.get(str(edge.get('source_node', '')), edge.get('source_node', ''))
                dst = node_name_map.get(str(edge.get('target_node', '')), edge.get('target_node', ''))
                adjacency.setdefault(src, []).append(dst)
            parts.append(f"# EDGES.{edge_type} ({len(type_edges)} relations):")
            for src, dsts in adjacency.items():
                parts.append(f"{src} -> [{', '.join(map(str, dsts))}]")
            parts.append("")

        return "\n".join(parts).strip()

    def _kg_level_2_summary(self, kg_data: dict, display_name_keys: list = None) -> str:
        """Render counts and up to five example names per type (level 2)."""
        data = kg_data.get("data") or {}
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        cypher_query = kg_data.get("cypher_query")

        if display_name_keys is None:
            display_name_keys = ['commonName', 'name', 'geneSymbol', 'symbol', 'title']

        parts = []
        if cypher_query:
            parts.append(f"CYPHER: {cypher_query}")
        parts.append(
            "[truncated: showing statistical summary because the full result set "
            "is too large for the current model's context window]"
        )

        nodes_by_type: dict = {}
        for node in nodes:
            readable = next((node.get(k) for k in display_name_keys if node.get(k)), None)
            nodes_by_type.setdefault(node.get('label', 'Entity'), []).append(readable)

        for node_type, names in nodes_by_type.items():
            samples = [n for n in names if n][:5]
            sample_str = ", ".join(map(str, samples)) if samples else "(no readable names)"
            parts.append(f"# NODES.{node_type}: {len(names)} entities. Top {len(samples)} examples: {sample_str}")

        edges_by_type: dict = {}
        for edge in edges:
            edges_by_type.setdefault(edge.get('label', 'RELATED_TO'), []).append(edge)

        for edge_type, type_edges in edges_by_type.items():
            sources = {str(e.get('source_node', '')) for e in type_edges}
            targets = {str(e.get('target_node', '')) for e in type_edges}
            parts.append(
                f"# EDGES.{edge_type}: {len(type_edges)} relations "
                f"between {len(sources)} distinct sources and {len(targets)} distinct targets"
            )

        return "\n".join(parts)

############################
# QUERY
############################

    def query(self, user_query: str, kg_mode: str = "Force", custom_system_prompt: Optional[str] = None, output_schema: Optional[Type[BaseModel]] = None, thread_id: Optional[str] = None) -> QueryResult:
        """Run a query to completion and return a QueryResult.

        Without ``thread_id`` each call starts a new thread; passing the same
        ``thread_id`` continues that conversation. Logs a warning when the
        checkpointer is persistent.
        """
        if not isinstance(self.checkpointer, MemorySaver):
            logger.warning("Executing a 'query' on an agent with a persistent checkpointer. A new thread will be created and saved.")

        thread_id = thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        stream_generator = self.stream_agent(
            user_query=user_query,
            thread_id=thread_id,
            kg_mode=kg_mode,
            custom_system_prompt=custom_system_prompt,
            output_schema=output_schema
        )

        thinking_steps = []
        for chunk in stream_generator:
             if chunk.get("type") == "updates" and chunk.get("thinking"):
                thinking_steps.append(chunk["thinking"])


        final_state_graph = self.checkpointer.get(config)
        messages = final_state_graph["channel_values"]["messages"]

        cypher_query = None
        cypher_first_attempt = None
        n_correction_iterations = None
        n_neo4j_error_corrections = None
        fallback_triggered = None
        kg_results = None
        data = None
        error = None
        final_answer = ""
        last_tool_message = next((msg for msg in reversed(messages) if isinstance(msg, ToolMessage)), None)

        if last_tool_message:
            try:
                content = json.loads(last_tool_message.content)
                if isinstance(content, dict):
                    cypher_query = content.get("cypher_query")
                    cypher_first_attempt = content.get("cypher_first_attempt")
                    n_correction_iterations = content.get("n_correction_iterations")
                    n_neo4j_error_corrections = content.get("n_neo4j_error_corrections")
                    fallback_triggered = content.get("fallback_triggered")
                    kg_results = content.get("data")
                    error = content.get("error")
                    if "message" in content:
                        thinking_steps.append(content["message"])
            except (json.JSONDecodeError, AttributeError):
                error = "Failed to parse tool output."
                logger.warning(f"Could not parse ToolMessage content: {last_tool_message.content}")

        last_ai_message = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
        if last_ai_message:
            final_answer = normalize_message_content(last_ai_message.content)

        if output_schema:
            try:
                data = json.loads(final_answer)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse structured output from AIMessage: {final_answer}")
                error = "Failed to parse structured output from LLM."

        return QueryResult(
            answer=final_answer,
            cypher_query=cypher_query,
            cypher_first_attempt=cypher_first_attempt,
            n_correction_iterations=n_correction_iterations,
            n_neo4j_error_corrections=n_neo4j_error_corrections,
            fallback_triggered=fallback_triggered,
            thinking=thinking_steps,
            data=data,
            kg_results=kg_results,
            error=error
        )


############################
# USER-REQUESTED STOP PERSISTENCE
############################

    STOP_MARKER = "_(Generation stopped by user.)_"

    def persist_stop_marker(self, thread_id: str, kg_mode: str = "Auto") -> None:
        """Append the user-stop message to the thread's saved state so it shows on reload."""
        try:
            graph = self.build_agent_graph(kg_mode=kg_mode)
            graph.update_state(
                {"configurable": {"thread_id": thread_id}},
                {"messages": [AIMessage(content=self.STOP_MARKER)]},
            )
        except Exception as e:
            logger.warning(
                f"persist_stop_marker failed for thread {thread_id}: {e}",
                exc_info=True,
            )

############################
# STREAM AGENT
############################

    def stream_agent(self, user_query: str, thread_id: str, kg_mode: str = "Auto", title: Optional[str] = None, custom_system_prompt: Optional[str] = None, output_schema: Optional[Type[BaseModel]] = None, output_schema_json: Optional[str] = None, output_schema_name: Optional[str] = None) -> Generator[Dict, None, None]:
        """Run the agent on ``thread_id`` and yield status updates and message chunks."""
        config = {"configurable": {"thread_id": thread_id}}

        # Raises UserStoppedException inside the KG pipeline when the user stops;
        # handled by agent_api.stream_generator.
        from nexa.api.stop_registry import make_stop_check
        _stop_check = make_stop_check(thread_id)

        agent_graph_instance = self.build_agent_graph(
            kg_mode=kg_mode, output_schema=output_schema, stop_check=_stop_check
        )

        initial_state_dict = {
            "messages": [HumanMessage(content=user_query)],
            "kg_profile_name": self._kg_profile_name_for_thread,
            "custom_system_prompt": custom_system_prompt,
            "structured_output_schema_json": output_schema_json,
            "structured_output_name": output_schema_name
        }
        if title is not None:
            initial_state_dict["title"] = title

        state = AgentState(**initial_state_dict)

        logger.info(f"Streaming agent for thread '{thread_id}' with LLM: {self._llm.__class__.__name__}, KG Mode: '{kg_mode}', KG Profile for thread: '{self._kg_profile_name_for_thread}'")

        for namespace, mode, chunk in agent_graph_instance.stream(state,
                                                               config=config,
                                                               stream_mode=["messages", "updates"],
                                                               subgraphs=True):
            if mode == "updates":
                if isinstance(chunk, dict):
                    for _, sub_dict in chunk.items():

                        if isinstance(sub_dict, dict):
                            status_obj = sub_dict.get('status')
                            status_val = status_obj.value if isinstance(status_obj, Status) else str(status_obj or "")
                            thinking = sub_dict.get('thinking', '') or ''

                            if len(namespace) == 2 and namespace[0].startswith('tools:') and (status_val or thinking):
                                logger.debug(f"Yielding update: NS={namespace}, Status={status_val}, Thinking={thinking}")
                                data = {
                                "type": "updates",
                                "status": status_val,
                                "thinking": thinking
                                }
                                yield data
                else:
                    logger.warning(f"Received 'updates' chunk is not a dict: {type(chunk)}")

            elif mode == "messages":
                 if isinstance(chunk, tuple) and len(chunk) == 2:
                     message_content, metadata = chunk
                     if isinstance(metadata, dict) and metadata.get('langgraph_node', '') == "agent" and hasattr(message_content, 'content') and message_content.content:
                         normalized_content = normalize_message_content(message_content.content)
                         if normalized_content:
                             data = {
                             "type": "messages",
                             "content": normalized_content
                             }
                             yield data
                 else:
                     logger.warning(f"Received 'messages' chunk is not a tuple of size 2: {chunk}")
            else:
                 logger.warning(f"Received chunk of unknown type: {mode}")

        logger.info(f"Finished streaming agent for thread '{thread_id}'")
