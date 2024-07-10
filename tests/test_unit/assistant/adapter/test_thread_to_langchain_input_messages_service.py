from unittest.mock import MagicMock
import pytest
from openai.types.beta.threads import Message
from langchain_openai_api_bridge.assistant.adapter.thread_to_langchain_input_messages_service import (
    ThreadToLangchainInputMessagesService,
)
from langchain_openai_api_bridge.assistant.adapter.openai_message_factory import (
    create_message,
)
from langchain_openai_api_bridge.assistant.repository.message_repository import (
    MessageRepository,
)
from openai.types.beta.threads import (
    TextContentBlock,
    ImageURLContentBlock,
    ImageURL,
    Text,
)


class TestRetreiveInputDict:

    @pytest.fixture
    def message_1(self) -> Message:
        return create_message(
            id="1", thread_id="1234", role="user", content="Hello, how are you?"
        )

    @pytest.fixture
    def multimodal_message(self) -> Message:
        return create_message(
            id="2",
            thread_id="4567",
            role="user",
            content=[
                TextContentBlock(
                    type="text", text=Text(value="What's in the image?", annotations=[])
                ),
                ImageURLContentBlock(
                    type="image_url",
                    image_url=ImageURL(url="https://example.com/image.png"),
                ),
            ],
        )

    @pytest.fixture
    def message_repository_mock(self, message_1, multimodal_message):
        message_repository_mock = MagicMock(spec=MessageRepository)

        def side_effect(*args, **kwargs):
            if kwargs["thread_id"] == "1234":
                return [message_1]
            elif kwargs["thread_id"] == "4567":
                return [multimodal_message]
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
        result = instance.retreive_input(thread_id="1234")

        assert len(result) == 1
        assert result[0].type == "human"
        assert result[0].content == [{"type": "text", "text": "Hello, how are you?"}]

    def test_image_url(self, instance: ThreadToLangchainInputMessagesService):
        result = instance.retreive_input(thread_id="4567")

        assert len(result) == 1
        assert result[0].type == "human"
        assert result[0].content[0] == {"type": "text", "text": "What's in the image?"}
        assert result[0].content[1] == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.png"},
        }
