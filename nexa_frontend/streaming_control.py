"""Stop button for an in-flight stream.

While `st.write_stream` runs, Streamlit processes no widget events, so a
regular button cannot stop the stream. Instead, a JS snippet replaces the
disabled chat input's send button with a Stop button that POSTs to
`/threads/{id}/stop` directly. Injected DOM nodes persist across reruns, so
the snippet is also rendered with `streaming=False` to remove the button.
"""

from __future__ import annotations

import os

import streamlit as st


_API_BASE_URL_FOR_BROWSER = os.getenv("BROWSER_API_URL", "http://localhost:8000")


def render_stop_button(thread_id: str, streaming: bool = True) -> None:
    """Mount or remove the Stop overlay on the chat input's send button.

    Args:
        thread_id: used to build the stop URL; None when no thread is active.
        streaming: True mounts the Stop button; False removes any leftover
            one and restores the send button.
    """
    stop_url = f"{_API_BASE_URL_FOR_BROWSER}/threads/{thread_id}/stop" if thread_id else ""
    streaming_flag = "true" if streaming else "false"
    st.html(
        f"""
        <script>
          (function() {{
            const streaming = {streaming_flag};
            const stopUrl = '{stop_url}';
            const STREAMLIT_RED = '#ff4b4b';
            const STREAMLIT_RED_HOVER = '#e23a3a';
            const doc = window.parent ? window.parent.document : document;

            function cleanup() {{
              const existing = doc.getElementById('nexa-stop-btn');
              if (existing) existing.remove();
              // Show the native send button again, on every chat_input
              // instance if more than one is mounted.
              doc.querySelectorAll('[data-testid="stChatInputSubmitButton"]').forEach(b => {{
                b.style.display = '';
              }});
            }}

            function buildStopButton() {{
              const btn = document.createElement('button');
              btn.id = 'nexa-stop-btn';
              btn.type = 'button';
              btn.innerText = '⏹';
              btn.title = 'Stop generation';
              btn.style.cssText = [
                'display:inline-flex',
                'align-items:center',
                'justify-content:center',
                'width:2rem',
                'height:2rem',
                'padding:0',
                'font-size:0.95rem',
                'font-weight:600',
                'line-height:1',
                'cursor:pointer',
                'border:none',
                'border-radius:0.4rem',
                'background-color:' + STREAMLIT_RED,
                'color:white',
                'transition:background-color 0.1s ease',
              ].join(';');
              btn.addEventListener('mouseenter', () => {{
                if (!btn.disabled) btn.style.backgroundColor = STREAMLIT_RED_HOVER;
              }});
              btn.addEventListener('mouseleave', () => {{
                if (!btn.disabled) btn.style.backgroundColor = STREAMLIT_RED;
              }});
              btn.addEventListener('click', async () => {{
                btn.disabled = true;
                btn.innerText = '⏳';
                btn.style.backgroundColor = '#9e9e9e';
                console.log('[nexa-stop] POST', stopUrl);
                try {{
                  const resp = await fetch(stopUrl, {{ method: 'POST', mode: 'cors' }});
                  console.log('[nexa-stop] response', resp.status);
                  if (!resp.ok) {{
                    btn.innerText = '⚠';
                    btn.style.backgroundColor = '#b71c1c';
                  }}
                }} catch (e) {{
                  console.error('[nexa-stop] fetch failed', e);
                  btn.innerText = '⚠';
                  btn.style.backgroundColor = '#b71c1c';
                }}
              }});
              return btn;
            }}

            function inject() {{
              // Idempotent: don't double-inject.
              if (doc.getElementById('nexa-stop-btn')) return true;
              const sendBtn = doc.querySelector('[data-testid="stChatInputSubmitButton"]');
              if (!sendBtn || !sendBtn.parentNode) return false;
              sendBtn.style.display = 'none';
              const stopBtn = buildStopButton();
              sendBtn.parentNode.insertBefore(stopBtn, sendBtn);
              return true;
            }}

            // Always clean up first: after a stream, the rerun finds the
            // previous turn's button still in the DOM.
            cleanup();

            if (streaming) {{
              if (!inject()) {{
                // chat_input may not be in the DOM yet on first paint.
                let attempts = 0;
                const iv = setInterval(() => {{
                  attempts++;
                  if (inject() || attempts > 40) clearInterval(iv);
                }}, 50);
              }}
            }}
          }})();
        </script>
        """,
        unsafe_allow_javascript=True,
    )
