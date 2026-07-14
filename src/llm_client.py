import os

def _get_llm():
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is not set")

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

def _response_content_to_text(content) -> str:
    """
    Приводит response.content к строке.

    У некоторых провайдеров (например, langchain_google_genai / Gemini)
    content бывает не строкой, а списком частей — строк или словарей вида
    {"type": "text", "text": "..."}. Эта функция нормализует оба
    варианта в обычную строку.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", ""))
        return "".join(parts)
    return str(content)