from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def build_ollama_model(model: str, base_url: str, temperature: float) -> ChatOllama:
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
    )


def build_lmstudio_model(model: str, base_url: str, temperature: float) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key="lm-studio",
        temperature=temperature,
    )
