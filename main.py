import html
import os

import streamlit as st
from dotenv import load_dotenv

from conversation import ConversationState, ModelConfig, get_active_speaker_name, run_single_turn

load_dotenv()

st.set_page_config(page_title="Dual Model Chat", page_icon="💬", layout="wide")

DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL")
DEFAULT_OLLAMA_NAME = os.getenv("OLLAMA_NAME")

DEFAULT_LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL")
DEFAULT_LMSTUDIO_URL = os.getenv("LMSTUDIO_BASE_URL")
DEFAULT_LMSTUDIO_NAME = os.getenv("LMSTUDIO_NAME")

DEFAULT_SEED = os.getenv("SEED_MESSAGE")
DEFAULT_MAX_TURNS = int(os.getenv("MAX_TURNS"))
DEFAULT_TEMP_A = float(os.getenv("TEMPERATURE_A"))
DEFAULT_TEMP_B = float(os.getenv("TEMPERATURE_B"))


def init_session_state() -> None:
    defaults = {
        "running": False,
        "messages": [],
        "turn_index": 0,
        "last_error": None,
        "conv_state": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def build_config_from_sidebar() -> ConversationState:
    return ConversationState(
        model_a=ModelConfig(
            name=st.session_state.ollama_name,
            model=st.session_state.ollama_model,
            base_url=st.session_state.ollama_url,
            system_prompt=st.session_state.ollama_system,
            temperature=st.session_state.temp_a,
        ),
        model_b=ModelConfig(
            name=st.session_state.lmstudio_name,
            model=st.session_state.lmstudio_model,
            base_url=st.session_state.lmstudio_url,
            system_prompt=st.session_state.lmstudio_system,
            temperature=st.session_state.temp_b,
        ),
        seed_message=st.session_state.seed_message,
        max_turns=st.session_state.max_turns,
        turn_index=st.session_state.turn_index,
        messages=st.session_state.messages,
        running=st.session_state.running,
        last_error=st.session_state.last_error,
    )


def sync_session_from_state(state: ConversationState) -> None:
    st.session_state.turn_index = state.turn_index
    st.session_state.messages = state.messages
    st.session_state.running = state.running
    st.session_state.last_error = state.last_error
    st.session_state.conv_state = state


def inject_bubble_styles() -> None:
    st.markdown(
        """<style>
    .chat-row { display: flex; margin: 0.4rem 0; }
    .chat-row.left { justify-content: flex-start; }
    .chat-row.right { justify-content: flex-end; }
    .bubble {
        max-width: 75%; padding: 0.65rem 0.9rem; border-radius: 1rem;
        line-height: 1.45; font-size: 0.95rem;
        color: #1a1a1a !important;
    }
    .bubble.left { background: #e8eef5; border-bottom-left-radius: 0.25rem; }
    .bubble.right { background: #d4edda; border-bottom-right-radius: 0.25rem; }
    .bubble-meta {
        font-size: 0.75rem; margin-bottom: 0.2rem;
        color: #555 !important; font-weight: 600;
    }
    .bubble.typing { opacity: 0.75; font-style: italic; color: #555 !important; }
    </style>""",
        unsafe_allow_html=True,
    )


def _format_bubble_content(content: str) -> str:
    return html.escape(content).replace("\n", "<br>")


def render_bubble(speaker: str, content: str, side: str, turn: int) -> None:
    align = "left" if side == "left" else "right"
    body = _format_bubble_content(content)
    st.markdown(
        f'<div class="chat-row {align}"><div class="bubble {align}" style="color:#1a1a1a;">'
        f'<div class="bubble-meta" style="color:#555;">{html.escape(speaker)} · Turn {turn + 1}</div>'
        f'<span style="color:#1a1a1a;">{body}</span></div></div>',
        unsafe_allow_html=True,
    )


def render_typing_bubble(speaker: str, side: str) -> None:
    align = "left" if side == "left" else "right"
    st.markdown(
        f'<div class="chat-row {align}"><div class="bubble {align} typing" style="color:#555;">'
        f'<div class="bubble-meta" style="color:#555;">{html.escape(speaker)}</div>'
        f"typing...</div></div>",
        unsafe_allow_html=True,
    )


init_session_state()

st.title("Dual Local Model Chat")
st.caption(f"Two local models chat with each other — {DEFAULT_OLLAMA_NAME} and {DEFAULT_LMSTUDIO_NAME} take turns.")

with st.sidebar:
    st.header("Model A")
    st.session_state.ollama_name = st.text_input(
        "Display name",
        value=DEFAULT_OLLAMA_NAME,
        key="input_ollama_name",
    )
    st.session_state.ollama_model = st.text_input(
        "Model name",
        value=DEFAULT_OLLAMA_MODEL,
        key="input_ollama_model",
    )
    st.session_state.ollama_url = st.text_input(
        "Base URL",
        value=DEFAULT_OLLAMA_URL,
        key="input_ollama_url",
    )
    st.session_state.ollama_system = st.text_area(
        "System prompt",
        value="You are a thoughtful AI participating in a conversation with another AI. Respond naturally and build on what the other said. Keep replies to at most 2 sentences.",
        key="input_ollama_system",
        height=100,
    )
    st.session_state.temp_a = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.5,
        value=DEFAULT_TEMP_A,
        step=0.1,
        key="input_temp_a",
    )

    st.divider()
    st.header("Model B")
    st.session_state.lmstudio_name = st.text_input(
        "Display name",
        value=DEFAULT_LMSTUDIO_NAME,
        key="input_lmstudio_name",
    )
    st.session_state.lmstudio_model = st.text_input(
        "Model name",
        value=DEFAULT_LMSTUDIO_MODEL,
        key="input_lmstudio_model",
    )
    st.session_state.lmstudio_url = st.text_input(
        "Base URL",
        value=DEFAULT_LMSTUDIO_URL,
        key="input_lmstudio_url",
    )
    st.session_state.lmstudio_system = st.text_area(
        "System prompt",
        value="You are a curious AI having a dialogue with another AI. Respond naturally and build on what the other said, challenge ideas, and keep the conversation engaging. Keep replies to at most 2 sentences.",
        key="input_lmstudio_system",
        height=100,
    )
    st.session_state.temp_b = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.5,
        value=DEFAULT_TEMP_B,
        step=0.1,
        key="input_temp_b",
    )

    st.divider()
    st.header("Conversation")
    st.session_state.seed_message = st.text_area(
        "Seed message (starts the chat)",
        value=DEFAULT_SEED,
        key="input_seed",
        height=80,
    )
    st.session_state.max_turns = st.number_input(
        "Max turns",
        min_value=2,
        max_value=200,
        value=DEFAULT_MAX_TURNS,
        step=2,
        key="input_max_turns",
    )

col_start, col_stop, col_clear = st.columns(3)

with col_start:
    start_clicked = st.button("Start", type="primary", use_container_width=True)
with col_stop:
    stop_clicked = st.button("Stop", use_container_width=True)
with col_clear:
    clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    st.session_state.running = False
    st.session_state.messages = []
    st.session_state.turn_index = 0
    st.session_state.last_error = None
    st.session_state.conv_state = None
    st.rerun()

if stop_clicked:
    st.session_state.running = False
    st.rerun()

seed_empty = not st.session_state.seed_message.strip()

if start_clicked:
    if seed_empty:
        st.error("Please enter a seed message before starting.")
    else:
        st.session_state.running = True
        st.session_state.messages = []
        st.session_state.turn_index = 0
        st.session_state.last_error = None
        st.rerun()

if seed_empty and not st.session_state.running:
    st.info("Enter a seed message in the sidebar, then click **Start**.")

status_col, _ = st.columns([2, 1])
with status_col:
    if st.session_state.running:
        active = get_active_speaker_name(build_config_from_sidebar())
        st.status(
            f"Turn {st.session_state.turn_index + 1}/{st.session_state.max_turns} — {active} is thinking...",
            state="running",
        )
    elif st.session_state.turn_index >= st.session_state.max_turns and st.session_state.messages:
        st.success(f"Reached max turns ({st.session_state.max_turns}). Conversation stopped.")
    elif st.session_state.messages:
        st.info("Conversation paused. Click **Start** to continue from the beginning, or **Clear** to reset.")

if st.session_state.last_error:
    st.error(st.session_state.last_error)

inject_bubble_styles()

for msg in st.session_state.messages:
    side = "left" if msg["turn"] % 2 == 0 else "right"
    render_bubble(msg["speaker"], msg["content"], side, msg["turn"])

if st.session_state.running and st.session_state.turn_index < st.session_state.max_turns:
    state = build_config_from_sidebar()
    active_name = get_active_speaker_name(state)
    typing_side = "left" if state.turn_index % 2 == 0 else "right"
    render_typing_bubble(active_name, typing_side)

    with st.spinner(f"{active_name} is thinking..."):
        state = run_single_turn(state)

    sync_session_from_state(state)

    if state.last_error:
        st.error(state.last_error)
    elif state.running and state.turn_index < state.max_turns:
        st.rerun()
