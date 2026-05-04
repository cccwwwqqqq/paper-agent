from __future__ import annotations

from agentic_rag.settings import Settings


def _create_openai_compatible_llm(model, temperature, api_key, base_url=None, provider_label="API"):
    if not api_key:
        raise ValueError(
            f"{provider_label} API key is missing. Set the required environment variable before starting the app."
        )

    from langchain_openai import ChatOpenAI

    kwargs = {
        "model": model,
        "temperature": temperature,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def create_llm(settings: Settings):
    provider = settings.llm_provider

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            base_url=settings.ollama_base_url,
        )

    if provider == "openai":
        return _create_openai_compatible_llm(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
            provider_label="OpenAI",
        )

    if provider == "deepseek":
        return _create_openai_compatible_llm(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            provider_label="DeepSeek",
        )

    if provider == "openai_compatible":
        return _create_openai_compatible_llm(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.openai_compat_api_key,
            base_url=settings.openai_compat_base_url,
            provider_label="OpenAI-compatible",
        )

    raise ValueError(
        "Unsupported LLM_PROVIDER '{}'. Use one of: ollama, openai, deepseek, openai_compatible.".format(provider)
    )

