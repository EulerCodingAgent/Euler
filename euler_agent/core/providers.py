"""Provider abstraction for cloud and local LLM backends."""

from __future__ import annotations

import os

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from euler_agent.config.settings import Provider


def get_chat_model(
    provider: Provider,
    model: str,
    api_key: str,
    base_url: str = "",
) -> BaseChatModel:
    if provider == "openai":
        os.environ["OPENAI_API_KEY"] = api_key
        return ChatOpenAI(model=model, temperature=0)
    if provider == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = api_key
        return ChatAnthropic(model=model, temperature=0)
    if provider == "ollama":
        # Ollama exposes an OpenAI-compatible endpoint at /v1.
        resolved_base = (base_url or "http://localhost:11434/v1").strip()
        return ChatOpenAI(
            model=model,
            temperature=0,
            base_url=resolved_base,
            api_key=api_key or "ollama",
        )
    if provider == "local":
        # Generic OpenAI-compatible local endpoint (LM Studio, vLLM, etc).
        if not base_url.strip():
            raise ValueError(
                "Local provider requires base_url. "
                "Set it via: Euler config set --provider local --model <model> --base-url <url>"
            )
        return ChatOpenAI(
            model=model,
            temperature=0,
            base_url=base_url.strip(),
            api_key=api_key or "local",
        )

    os.environ["GOOGLE_API_KEY"] = api_key
    return ChatGoogleGenerativeAI(model=model, temperature=0)
