from __future__ import annotations


def estimate_context_tokens(messages: list) -> int:
    import tiktoken

    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    return sum(len(encoding.encode(str(msg.content))) for msg in messages if hasattr(msg, "content") and msg.content)
