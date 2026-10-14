"""Render the prefixed chunks streamed by the API into Streamlit containers."""

import json

import streamlit as st


# Prefixes that start each piece of the API stream.
_CHUNK_PREFIXES = (
    "[UPDATES]",
    "[THINKING_COMPLETE]",
    "[THINKING]",
    "[MESSAGES]",
    "[TOOL_SUMMARY]",
    "[FINAL]",
    "[ERROR]",
    "[STOPPED]",
)


def split_stream_chunk(decoded: str) -> list[str]:
    """Split a raw HTTP chunk into its prefixed pieces, in order.

    One chunk can carry several pieces (e.g. a TOOL_SUMMARY between MESSAGES
    tokens in flex mode). Whitespace-only pieces are dropped.
    """
    if not decoded:
        return []

    # Find every prefix occurrence and slice between them.
    positions: list[tuple[int, str]] = []
    for prefix in _CHUNK_PREFIXES:
        start = 0
        while True:
            idx = decoded.find(prefix, start)
            if idx == -1:
                break
            positions.append((idx, prefix))
            start = idx + len(prefix)

    if not positions:
        # No prefix: continuation of a [MESSAGES] body.
        return [decoded] if decoded.strip() else []

    positions.sort(key=lambda p: p[0])

    pieces: list[str] = []
    # Text before the first prefix continues the previous chunk's piece.
    if positions[0][0] > 0:
        prefix_lead = decoded[: positions[0][0]]
        if prefix_lead.strip():
            pieces.append(prefix_lead)

    for i, (pos, _prefix) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(decoded)
        piece = decoded[pos:end]
        if piece.strip():
            pieces.append(piece)
    return pieces


class StreamlitCallback:
    """Render streamed status updates, thinking, messages and flex trace events."""

    def __init__(self, parent_container: st.delta_generator.DeltaGenerator):
        self.parent_container = parent_container
        self.history_entries = []

        self.history_created = False

        self.last_display_text = ""
        self.last_status = None

        self.thinking_buffer = ""
        self.think_container_created = False
        self.think_expander_collapsed = True

        # flex-mode buffers ----------------------------------------------
        self.trace_events: list[dict] = []
        self.final_payload: dict | None = None

        self.think_placeholder_main = self.parent_container.empty()
        self.history_placeholder_main = self.parent_container.empty()
        self.current_thought_placeholder_main = self.parent_container.empty()

    def _create_history_container(self):
        """Create the history container if it does not already exist."""
        if not self.history_created:
            with self.history_placeholder_main:
                self.history_container = st.expander("📚 History", expanded=False)
                self.history_placeholder = self.history_container.empty()
            self.history_created = True

    def _update_history_display(self):
        """Update the history display if there are entries."""
        if self.history_entries:
            self._create_history_container()
            full_history_text = "\n\n".join(self.history_entries)
            self.history_placeholder.markdown(full_history_text)

    def _create_current_thought_container(self, status_name: str):
        """Create the Current Thought container with the given status name."""
        self.current_thought_placeholder_main.empty()
        with self.current_thought_placeholder_main:
            self.current_thought_expander = st.expander(f"⏳ {status_name}", expanded=True)
            self.current_content_placeholder = self.current_thought_expander.empty()

    def _create_think_container(self, expanded: bool, content_to_display: str):
        """(Re)create the thinking container with the given state and content."""
        self.think_placeholder_main.empty()
        with self.think_placeholder_main:
            think_expander_ui = st.expander("💭 Thinking...", expanded=expanded)
            self.think_content_placeholder = think_expander_ui.empty()
            if content_to_display.strip():
                self.think_content_placeholder.markdown(f"<i>{content_to_display}</i>", unsafe_allow_html=True)
        self.think_container_created = True
        self.think_expander_collapsed = not expanded

    def _clear_think_container(self):
        """Remove the thinking container from the display."""
        self.think_placeholder_main.empty()
        self.think_container_created = False
        self.think_expander_collapsed = True

    def _update_think_display(self):
        """Update the thinking display based on self.thinking_buffer."""
        current_buffered_content_stripped = self.thinking_buffer.strip()

        if current_buffered_content_stripped:
            if not self.think_container_created or self.think_expander_collapsed:
                self._create_think_container(expanded=True, content_to_display=self.thinking_buffer)
            else:
                self.think_content_placeholder.markdown(f"<i>{self.thinking_buffer}</i>", unsafe_allow_html=True)
        else:
            if self.think_container_created:
                self._clear_think_container()

    def _finalize_thinking(self):
        """Finalize the thinking session. Show the expander collapsed if content exists, otherwise remove it."""
        final_buffered_text_stripped = self.thinking_buffer.strip()

        if final_buffered_text_stripped:
            self._create_think_container(expanded=False, content_to_display=self.thinking_buffer)
        else:
            if self.think_container_created:
                self._clear_think_container()

        self.thinking_buffer = ""

    def process_chunk(self, chunk: str):
        """Process one HTTP chunk and return the message text it carries, or None."""
        pieces = split_stream_chunk(chunk)
        if not pieces:
            return None

        out_parts: list[str] = []
        for piece in pieces:
            text = self._process_single_piece(piece)
            if text:
                out_parts.append(text)
        if not out_parts:
            return None
        return "".join(out_parts)

    def _process_single_piece(self, piece: str):
        if piece.startswith("[UPDATES]"):
            lines = piece.split("\n")
            status_line = lines[1] if len(lines) > 1 else "Unknown status"
            thinking_line = "\n".join(lines[2:]).rstrip("\n") if len(lines) > 2 else "Unknown thinking"
            self.update(status=status_line, thinking=thinking_line)
            return None

        if piece.startswith("[THINKING_COMPLETE]"):
            self._finalize_thinking()
            return None

        if piece.startswith("[THINKING]"):
            self.thinking_buffer += piece[len("[THINKING]"):]
            self._update_think_display()
            return None

        if piece.startswith("[MESSAGES]"):
            return piece[len("[MESSAGES]"):]

        if piece.startswith("[TOOL_SUMMARY]"):
            self._record_tool_summary(piece[len("[TOOL_SUMMARY]"):])
            return None

        if piece.startswith("[FINAL]"):
            self._record_final(piece[len("[FINAL]"):])
            return None

        if piece.startswith("[ERROR]"):
            # Error raised server-side during the stream.
            err_text = piece[len("[ERROR]"):].strip()
            st.error(err_text)
            return None

        if piece.startswith("[STOPPED]"):
            # User stop: the persisted stop marker is rendered after the rerun.
            return None

        # Unrecognized piece: usually the continuation of a [MESSAGES] body.
        return piece

    # flex helpers -------------------------------------------------------------

    def _record_tool_summary(self, payload_text: str):
        try:
            event = json.loads(payload_text.strip())
        except json.JSONDecodeError:
            return
        if not isinstance(event, dict):
            return
        self.trace_events.append(event)
        # Also keep it in session state so it survives the rerun.
        st.session_state.setdefault("current_trace", []).append(event)
        # Show the new event in the live trace right away.
        self._live_update_trace_panel()

    def _record_final(self, payload_text: str):
        try:
            payload = json.loads(payload_text.strip())
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return
        self.final_payload = payload
        st.session_state["last_final"] = payload

    def _live_update_trace_panel(self) -> None:
        """Re-render the live trace into the placeholder set by `handle_streaming`.

        No-op without a placeholder (direct mode). Renders only the current
        turn's events, without eye buttons; the full panel returns after the rerun.
        """
        placeholder = st.session_state.get("_flex_trace_placeholder")
        if placeholder is None:
            return
        try:
            from nexa_frontend.chat_flex.trace import render_agent_trace_flex
            with placeholder.container():
                render_agent_trace_flex(live_only=True, empty_caption="")
        except Exception:
            # A failed live render must not break the stream.
            pass

    def update(self, status: str = None, thinking: str = None):
        """Update the display: archive the previous step to history and show the current step."""
        status = status or ""
        thinking = thinking or ""

        if self.last_status or self.last_display_text:
            history_lines = []
            if self.last_status:
                history_lines.append(f"**{self.last_status}**  ")
            if self.last_display_text:
                history_lines.append(f"{self.last_display_text}")

            history_entry = "\n".join(history_lines)
            if history_entry:
                self.history_entries.append(history_entry)
                self._update_history_display()

        current_display_text = thinking
        expander_title = status if status else "Current Thought"

        self._create_current_thought_container(expander_title)
        if current_display_text.strip():
            self.current_content_placeholder.markdown(current_display_text)
        else:
            self.current_content_placeholder.empty()

        self.last_display_text = current_display_text
        self.last_status = status

    def complete(self):
        """Finalize the process: archive the last step to history."""
        if self.thinking_buffer.strip():
            self._finalize_thinking()

        if self.last_status or self.last_display_text:
            history_lines = []
            if self.last_status:
                history_lines.append(f"**{self.last_status}**")
            if self.last_display_text:
                history_lines.append(f"{self.last_display_text}")

            history_entry = "\n".join(history_lines)
            if history_entry:
                self.history_entries.append(history_entry)
                self._update_history_display()

        self.last_display_text = ""
        self.last_status = None

    def get_final_history(self) -> list[str]:
        """Return the list of accumulated history entries."""
        return self.history_entries
