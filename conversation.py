from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from models import build_lmstudio_model, build_ollama_model

BREVITY_INSTRUCTION = (
    "Keep every reply to at most 2 sentences. "
    "Be concise and do not use bullet points or numbered lists."
)


@dataclass
class ModelConfig:
    name: str
    model: str
    base_url: str
    system_prompt: str
    temperature: float


@dataclass
class ConversationState:
    model_a: ModelConfig
    model_b: ModelConfig
    seed_message: str
    max_turns: int
    turn_index: int = 0
    messages: list[dict] = field(default_factory=list)
    running: bool = False
    last_error: str | None = None


def _get_input_text(state: ConversationState) -> str:
    if state.turn_index == 0:
        return state.seed_message
    return state.messages[-1]["content"]


def _get_active_config(state: ConversationState) -> ModelConfig:
    if state.turn_index % 2 == 0:
        return state.model_a
    return state.model_b


def _build_messages(config: ModelConfig, input_text: str) -> list:
    system_parts = []
    if config.system_prompt.strip():
        system_parts.append(config.system_prompt.strip())
    system_parts.append(BREVITY_INSTRUCTION)
    messages = [SystemMessage(content="\n\n".join(system_parts))]
    messages.append(HumanMessage(content=input_text))
    return messages


def run_single_turn(state: ConversationState) -> ConversationState:
    if state.turn_index >= state.max_turns:
        state.running = False
        return state

    config = _get_active_config(state)
    input_text = _get_input_text(state)

    try:
        if state.turn_index % 2 == 0:
            llm = build_ollama_model(
                model=config.model,
                base_url=config.base_url,
                temperature=config.temperature,
            )
        else:
            llm = build_lmstudio_model(
                model=config.model,
                base_url=config.base_url,
                temperature=config.temperature,
            )

        response = llm.invoke(_build_messages(config, input_text))
        content = response.content if isinstance(response, AIMessage) else str(response.content)

        state.messages.append(
            {
                "speaker": config.name,
                "content": content,
                "turn": state.turn_index,
            }
        )
        state.turn_index += 1
        state.last_error = None

        if state.turn_index >= state.max_turns:
            state.running = False

    except Exception as exc:
        state.last_error = f"{config.name} failed: {exc}"
        state.running = False

    return state


def get_active_speaker_name(state: ConversationState) -> str:
    return _get_active_config(state).name
