"""Provider abstraction for OpenAI, Anthropic, and Gemini."""

from __future__ import annotations

import os

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from euler_agent.config.settings import Provider


def get_chat_model(provider: Provider, model: str, api_key: str) -> BaseChatModel:
    if provider == "openai":
        os.environ["OPENAI_API_KEY"] = api_key
        return ChatOpenAI(model=model, temperature=0)
    if provider == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = api_key
        return ChatAnthropic(model=model, temperature=0)

    os.environ["GOOGLE_API_KEY"] = api_key
    return ChatGoogleGenerativeAI(model=model, temperature=0)
