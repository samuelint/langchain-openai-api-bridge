import logging
import pytest
from openai import OpenAI

from fastapi.testclient import TestClient
import validators
from assistant_server_openai import api


test_api = TestClient(api)


logging.getLogger("openai").setLevel(logging.DEBUG)


@pytest.fixture
def openai_client():
    return OpenAI(
        base_url="http://testserver/my-assistant/openai/v1",
        http_client=test_api,
    )


class TestThread:
    def test_create_empty_thread(self, openai_client: OpenAI):
        thread = openai_client.beta.threads.create()

        assert validators.uuid(thread.id)

    def test_delete_thread(self, openai_client: OpenAI):
        created_thread = openai_client.beta.threads.create()

        openai_client.beta.threads.delete(thread_id=created_thread.id)
        deleted_thread = openai_client.beta.threads.retrieve(
            thread_id=created_thread.id
        )

        assert deleted_thread is None

    def test_create_thread_with_metadata(self, openai_client: OpenAI):
        thread = openai_client.beta.threads.create(metadata={"key": "value"})

        assert thread.metadata["key"] == "value"

    def test_retreive_thread_with_metadata(self, openai_client: OpenAI):
        created_thread = openai_client.beta.threads.create(metadata={"key": "value"})

        retreived_thread = openai_client.beta.threads.retrieve(
            thread_id=created_thread.id
        )

        assert retreived_thread.metadata["key"] == "value"


class TestMessage:
    def test_create_thread_with_messages(self, openai_client: OpenAI):
        created_thread = openai_client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": "Hello, what is AI?",
                },
                {
                    "role": "user",
                    "content": "How does AI work? Explain it in simple terms.",
                },
            ]
        )

        messages = openai_client.beta.threads.messages.list(thread_id=created_thread.id)

        assert len(messages.data) == 2
        assert messages.data[0].role == "user"
        assert messages.data[0].content[0].type == "text"
        assert messages.data[0].content[0].text.value == "Hello, what is AI?"
        assert messages.data[1].role == "user"
        assert messages.data[1].content[0].type == "text"
        assert (
            messages.data[1].content[0].text.value
            == "How does AI work? Explain it in simple terms."
        )

    def test_retreive_message(self, openai_client: OpenAI):
        created_thread = openai_client.beta.threads.create()
        created_message = openai_client.beta.threads.messages.create(
            thread_id=created_thread.id, role="user", content="Hello, what is AI?"
        )

        message = openai_client.beta.threads.messages.retrieve(
            message_id=created_message.id, thread_id=created_thread.id
        )

        assert message.role == "user"
        assert message.content[0].type == "text"
        assert message.content[0].text.value == "Hello, what is AI?"

    def test_delete_message(self, openai_client: OpenAI):
        created_thread = openai_client.beta.threads.create()
        created_message = openai_client.beta.threads.messages.create(
            thread_id=created_thread.id, role="user", content="Hello, what is AI?"
        )

        created_message = openai_client.beta.threads.messages.delete(
            thread_id=created_thread.id, message_id=created_message.id
        )
        message = openai_client.beta.threads.messages.retrieve(
            message_id=created_message.id, thread_id=created_thread.id
        )

        assert message is None
