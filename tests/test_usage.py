from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms import ChatResponse

from mobilerun.agent.usage import get_usage_from_response


def test_ollama_llm_class_name_usage_is_supported():
    response = ChatResponse(
        message=ChatMessage(role="assistant", content="ok"),
        raw={"prompt_eval_count": 12, "eval_count": 5},
    )

    usage = get_usage_from_response("Ollama_llm", response)

    assert usage.request_tokens == 12
    assert usage.response_tokens == 5
    assert usage.total_tokens == 17
    assert usage.requests == 1
