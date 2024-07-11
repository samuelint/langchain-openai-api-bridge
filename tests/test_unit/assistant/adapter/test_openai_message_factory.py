from langchain_openai_api_bridge.assistant.adapter.openai_message_factory import (
    create_message_content,
    create_text_message_delta,
)


class TestCreateMessageContent:
    def test_object_content(self):
        result = create_message_content(
            content=[{"type": "text", "text": "test"}],
        )

        assert result[0].text.value == "test"
        assert result[0].type == "text"

    def test_string_content(self):
        result = create_message_content(content="test")

        assert result[0].text.value == "test"

    def test_given_string_content_type_is_text(self):
        result = create_message_content(content="test")

        assert result[0].type == "text"

    def test_given_string_content_and_no_annotations_then_annotations_are_empty_array(
        self,
    ):
        result = create_message_content(content="test")

        assert result[0].text.annotations == []


class TestCreateTextMessageDelta:

    def test_object_content(self):
        result = create_text_message_delta(
            role="assistant",
            content=[{"index": 0, "type": "text", "text": "test"}],
        )

        assert result.content[0].text.value == "test"
        assert result.content[0].type == "text"
        assert result.content[0].index == 0

    def test_string_content(self):
        result = create_text_message_delta(role="assistant", content="test")

        assert result.content[0].text.value == "test"
        assert result.content[0].type == "text"
        assert result.content[0].index == 0

    def test_role(self):
        result = create_text_message_delta(role="assistant", content="test")

        assert result.role == "assistant"
