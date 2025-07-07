from unittest.mock import patch
from langchain_core.runnables.schema import StandardStreamEvent

from langchain_openai_api_bridge.chat_completion.chat_completion_chunk_choice_adapter import (
    to_openai_chat_completion_chunk_choice,
    to_openai_chat_completion_chunk_object,
    to_openai_chat_message,
)


class FixtureEventChunk:
    def __init__(self, content: str, tool_call_chunks: list = []):
        self.content = content
        self.tool_call_chunks = tool_call_chunks


class TestToChatMessage:
    def test_message_have_content(self):
        event = StandardStreamEvent(
            data={"chunk": FixtureEventChunk(content="some content")}
        )

        result = to_openai_chat_message(event)

        assert result.content == "some content"

    def test_message_have_args_role(self):
        event = StandardStreamEvent(
            data={"chunk": FixtureEventChunk(content="some content")}
        )

        result = to_openai_chat_message(event, role="assistant")

        assert result.role == "assistant"

    def test_message_have_assistant_role_by_default(self):
        event = StandardStreamEvent(
            data={"chunk": FixtureEventChunk(content="some content")}
        )

        result = to_openai_chat_message(event)

        assert result.role == "assistant"


class TestToCompletionChunkChoice:
    def test_choice_finish_reason_is_none_by_default(self):
        event = StandardStreamEvent(
            data={"chunk": FixtureEventChunk(content="some content")}
        )

        result = to_openai_chat_completion_chunk_choice(event)

        assert result.finish_reason is None

    def test_choice_finish_reason_is_same_as_provided(self):
        event = StandardStreamEvent(
            data={"chunk": FixtureEventChunk(content="some content")}
        )

        result = to_openai_chat_completion_chunk_choice(event, finish_reason="stop")

        assert result.finish_reason == "stop"

    def test_choice_index_0_by_default(self):
        event = StandardStreamEvent(
            data={"chunk": FixtureEventChunk(content="some content")}
        )

        result = to_openai_chat_completion_chunk_choice(event)

        assert result.index == 0

    def test_choice_index_is_same_as_provided(self):
        event = StandardStreamEvent(
            data={"chunk": FixtureEventChunk(content="some content")}
        )

        result = to_openai_chat_completion_chunk_choice(event, index=69)

        assert result.index == 69

    def test_delta_message_have_content(self):
        event = StandardStreamEvent(
            data={"chunk": FixtureEventChunk(content="some content")}
        )

        result = to_openai_chat_completion_chunk_choice(event)

        assert result.delta.content == "some content"

    def test_delta_message_have_args_role(self):
        event = StandardStreamEvent(
            data={"chunk": FixtureEventChunk(content="some content")}
        )

        result = to_openai_chat_completion_chunk_choice(event, role="assistant")

        assert result.delta.role == "assistant"

    def test_delta_message_have_none_role_by_default(self):
        event = StandardStreamEvent(
            data={"chunk": FixtureEventChunk(content="some content")}
        )

        result = to_openai_chat_completion_chunk_choice(event)

        assert result.delta.role is None


class TestToCompletionChunkObject:
    def test_id_is_same_as_provider(self):
        event = StandardStreamEvent(
            data={"chunk": FixtureEventChunk(content="some content")}
        )

        result = to_openai_chat_completion_chunk_object(event, id="a")

        assert result.id == "a"

    def test_id_is_empty_string_when_not_defined_as_argument(self):
        event = StandardStreamEvent(
            data={"chunk": FixtureEventChunk(content="some content")}
        )

        result = to_openai_chat_completion_chunk_object(event)

        assert result.id == ""

    def test_id_empty_string_when_not_defined(self):
        event = StandardStreamEvent(
            data={"chunk": FixtureEventChunk(content="some content")}
        )

        result = to_openai_chat_completion_chunk_object(event)

        assert result.id == ""

    def test_object_is_always_chat_completion_chunk(self):
        event = StandardStreamEvent(
            data={"chunk": FixtureEventChunk(content="some content")}
        )

        result = to_openai_chat_completion_chunk_object(event)

        assert result.object == "chat.completion.chunk"

    @patch("time.time", return_value=1638316800)
    def test_created_is_current_time(self, mock_time):
        event = StandardStreamEvent(
            data={"chunk": FixtureEventChunk(content="some content")}
        )

        result = to_openai_chat_completion_chunk_object(event)

        assert result.created == 1638316800

    def test_choice_finish_reason_is_none_by_default(self):
        event = StandardStreamEvent(
            data={"chunk": FixtureEventChunk(content="some content")}
        )

        result = to_openai_chat_completion_chunk_object(event)

        assert result.choices[0].finish_reason is None

    def test_choice_finish_reason_is_same_as_provided(self):
        event = StandardStreamEvent(
            data={"chunk": FixtureEventChunk(content="some content")}
        )

        result = to_openai_chat_completion_chunk_object(event, finish_reason="stop")

        assert result.choices[0].finish_reason == "stop"

    def test_choice_index_0_by_default(self):
        event = StandardStreamEvent(
            data={"chunk": FixtureEventChunk(content="some content")}
        )

        result = to_openai_chat_completion_chunk_object(event)

        assert result.choices[0].index == 0

    def test_delta_message_have_content(self):
        event = StandardStreamEvent(
            data={"chunk": FixtureEventChunk(content="some content")}
        )

        result = to_openai_chat_completion_chunk_object(event)

        assert result.choices[0].delta.content == "some content"

    def test_delta_message_have_args_role(self):
        event = StandardStreamEvent(
            data={"chunk": FixtureEventChunk(content="some content")}
        )

        result = to_openai_chat_completion_chunk_object(event, role="assistant")

        assert result.choices[0].delta.role == "assistant"

    def test_delta_message_have_none_role_by_default(self):
        event = StandardStreamEvent(
            data={"chunk": FixtureEventChunk(content="some content")}
        )

        result = to_openai_chat_completion_chunk_object(event)

        assert result.choices[0].delta.role is None

    def test_delta_message_have_function_call(self):
        event = StandardStreamEvent(
            data={
                "chunk": FixtureEventChunk(
                    content="",
                    tool_call_chunks=[{"name": "my_func", "args": '{"x": 1}'}],
                )
            }
        )

        result = to_openai_chat_completion_chunk_object(event)

        assert result.choices[0].delta.function_call is not None
        assert result.choices[0].delta.function_call.name == "my_func"
