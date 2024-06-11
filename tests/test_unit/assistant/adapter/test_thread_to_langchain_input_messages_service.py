from unittest.mock import MagicMock
import pytest
from openai.types.beta.threads import Message
from langchain_openai_api_bridge.assistant.adapter.thread_to_langchain_input_messages_service import (
    ThreadToLangchainInputMessagesService,
)
from langchain_openai_api_bridge.assistant.openai_message_factory import create_message
from langchain_openai_api_bridge.assistant.repository.assistant_message_repository import (
    AssistantMessageRepository,
)


class TestRetreiveInputDict:

    @pytest.fixture
    def message_1(self) -> Message:
        return create_message(
            id="1", thread_id="1234", role="user", content="Hello, how are you?"
        )

    @pytest.fixture
    def message_repository_mock(self, message_1):
        message_repository_mock = MagicMock(spec=AssistantMessageRepository)

        def side_effect(*args, **kwargs):
            if kwargs["thread_id"] == "1234":
                return [message_1]
            else:
                return []

        message_repository_mock.list.side_effect = side_effect
        return message_repository_mock

    @pytest.fixture
    def instance(self, message_repository_mock):
        return ThreadToLangchainInputMessagesService(
            message_repository=message_repository_mock
        )

    def test_retreive_thread_messages(
        self, instance: ThreadToLangchainInputMessagesService
    ):
        result = instance.retreive_input_dict(thread_id="1234")

        assert result["messages"] == [
            {"role": "user", "content": "Hello, how are you?"}
        ]
