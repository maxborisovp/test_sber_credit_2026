import os
import pytest
from src.llm_client import _get_llm, _response_content_to_text

class TestGetLlm:
    def test_raises_without_api_key(self):
        os.environ.pop("GOOGLE_API_KEY", None)
        with pytest.raises(RuntimeError):
            _get_llm()

    def test_returns_client_with_api_key(self):
        os.environ["GOOGLE_API_KEY"] = "test_key"
        assert _get_llm() is not None

class TestResponseContentToText:
    def test_response_content_to_text_with_string(self):
        """Если content — строка, возвращается она же."""
        content = "Hello, world!"
        assert _response_content_to_text(content) == "Hello, world!"


    def test_response_content_to_text_with_list_of_strings(self):
        """Если content — список строк, они склеиваются."""
        content = ["Hello", ", ", "world", "!"]
        assert _response_content_to_text(content) == "Hello, world!"


    def test_response_content_to_text_with_list_of_dicts(self):
        """Если content — список словарей с ключом 'text', значения склеиваются."""
        content = [{"type": "text", "text": "Hello"}, {"type": "text", "text": ", world!"}]
        assert _response_content_to_text(content) == "Hello, world!"
    
    def test_response_content_to_text_with_mixed_list(self):
        """Список может содержать строки и словари — всё обрабатывается."""
        content = ["Start: ", {"type": "text", "text": "middle"}, " end"]
        assert _response_content_to_text(content) == "Start: middle end"
    
    def test_response_content_to_text_with_empty_list(self):
        """Пустой список возвращает пустую строку."""
        assert _response_content_to_text([]) == ""