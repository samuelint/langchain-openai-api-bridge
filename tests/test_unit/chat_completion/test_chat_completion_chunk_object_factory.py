from langchain_openai_api_bridge.chat_completion.chat_completion_chunk_object_factory import (
    create_final_chat_completion_chunk_object,
)


class TestCreateFinalChatCompletionChunkObject:
    def test_choice_final_reason_is_stop(self):
        result = create_final_chat_completion_chunk_object(id="a")

        assert result.choices[0].finish_reason == "stop"

    def test_id_is_mandatory(self):
        result = create_final_chat_completion_chunk_object(id="a")

        assert result.id == "a"

    def test_model_same(self):
        result = create_final_chat_completion_chunk_object(id="a", model="aaa")

        assert result.model == "aaa"

    def test_model_is_empty_string_when_not_provided(self):
        result = create_final_chat_completion_chunk_object(id="a")

        assert result.model == ""

    def test_system_fingerprint_is_empty_when_not_provided(self):
        result = create_final_chat_completion_chunk_object(id="a")

        assert result.system_fingerprint is None

    def test_system_fingerprint_is_used_when_provided(self):
        result = create_final_chat_completion_chunk_object(
            id="a", system_fingerprint="bbb"
        )

        assert result.system_fingerprint == "bbb"

    def test_final_chunk_finish_reason_tool_calls(self):
        chunk_obj = create_final_chat_completion_chunk_object(
            id="a",
            finish_reason="tool_calls",
        )

        assert chunk_obj.choices[0].finish_reason == "tool_calls"

    def test_final_chunk_finish_reason_stop(self):
        chunk_obj = create_final_chat_completion_chunk_object(
            id="a",
            finish_reason="stop",
        )

        assert chunk_obj.choices[0].finish_reason == "stop"
