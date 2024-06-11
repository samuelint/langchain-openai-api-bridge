import logging
from typing import List
import pytest
from openai import OpenAI
from openai.types.beta import AssistantStreamEvent

from fastapi.testclient import TestClient
import validators
from assistant_server_openai import api


test_api = TestClient(api)


logging.getLogger("openai").setLevel(logging.DEBUG)


@pytest.fixture(scope="session")
def openai_client():
    return OpenAI(
        base_url="http://testserver/my-assistant/openai/v1",
        http_client=test_api,
    )


class TestRunStream:

    @pytest.fixture(scope="session")
    def stream_response_events(
        self, openai_client: OpenAI
    ) -> List[AssistantStreamEvent]:
        thread = openai_client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": 'Say: "This is a test message."',
                },
            ]
        )

        stream = openai_client.beta.threads.runs.create(
            thread_id=thread.id,
            model="gpt-3.5-turbo",
            assistant_id="any",
            stream=True,
        )

        events: List[AssistantStreamEvent] = []
        for event in stream:
            events.append(event)

        return events

    @pytest.fixture
    def shared_stream_response_events(self, request) -> List[AssistantStreamEvent]:
        return request.config.cache.get("stream_response_events", [])

    def test_run_stream_starts_with_thread_run_created(
        self, stream_response_events: List[AssistantStreamEvent]
    ):
        assert stream_response_events[0].event == "thread.run.created"

    def test_run_stream_ends_with_thread_run_completed(
        self, stream_response_events: List[AssistantStreamEvent]
    ):
        assert stream_response_events[-1].event == "thread.run.completed"

    def test_run_stream_message_deltas(
        self, stream_response_events: List[AssistantStreamEvent]
    ):
        str_response = ""
        for event in stream_response_events:
            if event.event == "thread.message.delta":
                str_response += "".join(
                    event.data.data["delta"]["content"][0]["text"]["value"]
                )

        assert "This is a test message." in str_response


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
