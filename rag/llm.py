import json
import urllib.request

from langchain_openai import ChatOpenAI

from .config import LLM_BASE_URL, LLM_MODEL

_cache = {}


def model_name() -> str:
    if LLM_MODEL:
        return LLM_MODEL
    with urllib.request.urlopen(LLM_BASE_URL + "/models", timeout=10) as r:
        return json.load(r)["data"][0]["id"]


def get_llm(temperature: float = 0.0, max_tokens: int = 400) -> ChatOpenAI:
    """LLM local open weights servi par vLLM, via LangChain (mode « thinking » désactivé)."""
    key = (temperature, max_tokens)
    if key not in _cache:
        _cache[key] = ChatOpenAI(
            model=model_name(), base_url=LLM_BASE_URL, api_key="EMPTY",
            temperature=temperature, max_tokens=max_tokens, timeout=180,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return _cache[key]
