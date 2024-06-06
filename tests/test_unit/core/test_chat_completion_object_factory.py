from unittest.mock import patch

from langchain_openai_api_bridge.core.chat_completion_object_factory import (
    ChatCompletionObjectFactory,
)
from langchain_openai_api_bridge.core.types.openai import (
    OpenAIChatCompletionChoice,
    OpenAIChatCompletionUsage,
    OpenAIChatMessage,
)


class TestChatCompletionObjectFactory:

    def test_default_object(self):
        result = ChatCompletionObjectFactory.create(
            id="test",
            model="test",
        )

        assert result.object == "chat.completion"

    def test_default_choice(self):
        result = ChatCompletionObjectFactory.create(
            id="test",
            model="test",
        )

        assert len(result.choices) == 0

    def test_id(self):
        result = ChatCompletionObjectFactory.create(
            id="test-id",
            model="test",
        )

        assert result.id == "test-id"

    def test_model(self):
        result = ChatCompletionObjectFactory.create(
            id="test",
            model="test-model",
        )

        assert result.model == "test-model"

    @patch(
        "langchain_openai_api_bridge.core.chat_completion_object_factory.time.time",
        return_value=1638316800,
    )
    def test_created_is_current_time(self, mock_time):
        result = ChatCompletionObjectFactory.create(
            id="test",
            model="test-model",
        )

        assert result.created == 1638316800

    def test_created_with_argument_time(self):
        result = ChatCompletionObjectFactory.create(
            id="test",
            model="test-model",
            created=123,
        )

        assert result.created == 123

    def test_usage(self):
        result = ChatCompletionObjectFactory.create(
            id="test",
            model="test-model",
            usage=OpenAIChatCompletionUsage(
                total_tokens=100,
                prompt_tokens=50,
                completion_tokens=50,
            ),
        )

        assert result.usage.total_tokens == 100
        assert result.usage.prompt_tokens == 50
        assert result.usage.completion_tokens == 50

    def test_default_usage(self):
        result = ChatCompletionObjectFactory.create(
            id="test",
            model="test-model",
        )

        assert result.usage.total_tokens == -1
        assert result.usage.prompt_tokens == -1
        assert result.usage.completion_tokens == -1

    def test_messages(self):
        result = ChatCompletionObjectFactory.create(
            id="test",
            model="test-model",
            choices=[
                OpenAIChatCompletionChoice(
                    index=0,
                    message=OpenAIChatMessage(
                        role="assistant", content="test-message-assistant"
                    ),
                ),
            ],
        )

        assert result.choices[0].message.role == "assistant"
        assert result.choices[0].message.content == "test-message-assistant"

    def test_system_fingerprint(self):
        result = ChatCompletionObjectFactory.create(
            id="test",
            model="test-model",
            system_fingerprint="test-fingerprint",
        )

        assert result.system_fingerprint == "test-fingerprint"
