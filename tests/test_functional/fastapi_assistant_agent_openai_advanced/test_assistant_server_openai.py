import logging
from typing import List
import pytest
from openai import OpenAI
from openai.types.beta import AssistantStreamEvent, Thread

from fastapi.testclient import TestClient
import validators
from assistant_server_openai import api
from tests.test_functional.assistant_stream_utils import (
    assistant_stream_events_to_str_response,
)


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
    def thread(self, openai_client: OpenAI) -> Thread:
        return openai_client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": 'Say: "This is a test message."',
                },
            ]
        )

    @pytest.fixture(scope="session")
    def stream_response_events(
        self, openai_client: OpenAI, thread: Thread
    ) -> List[AssistantStreamEvent]:

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
        str_response = assistant_stream_events_to_str_response(stream_response_events)

        assert "This is a test message." in str_response

    def test_run_stream_message_delta_annotation_is_not_null(
        self, stream_response_events: List[AssistantStreamEvent]
    ):
        message_delta_events = [
            event
            for event in stream_response_events
            if event.event == "thread.message.delta"
        ]

        for event in message_delta_events:
            assert event.data.delta.content[0].text.annotations is not None

    def test_message_id_is_same_for_start_delta_ends(
        self, stream_response_events: List[AssistantStreamEvent]
    ):
        first_thread_message_completed = next(
            (
                event
                for event in stream_response_events
                if event.event == "thread.message.completed"
            ),
            None,
        )

        assert stream_response_events[1].event == "thread.message.created"
        assert stream_response_events[2].event == "thread.message.delta"
        assert first_thread_message_completed.event == "thread.message.completed"
        assert stream_response_events[1].data.id == stream_response_events[2].data.id
        assert (
            first_thread_message_completed.data.id == stream_response_events[2].data.id
        )

    def test_streamed_response_message_is_persisted(
        self,
        stream_response_events: List[AssistantStreamEvent],
        openai_client: OpenAI,
        thread: Thread,
    ):

        messages = openai_client.beta.threads.messages.list(thread_id=thread.id).data
        last_message = messages[-1]

        assert last_message.role == "assistant"
        assert last_message.status == "completed"
        assert len(last_message.content[0].text.value) > 0


class TestFollowupMessage:

    @pytest.fixture(scope="session")
    def thread(self, openai_client: OpenAI) -> Thread:
        return openai_client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": "My favorite fruit banana",
                },
            ]
        )

    def test_run_stream_starts_with_thread_run_created(
        self, openai_client: OpenAI, thread: Thread
    ):
        openai_client.beta.threads.runs.create(
            thread_id=thread.id,
            model="gpt-3.5-turbo",
            assistant_id="any",
            temperature=0,
            stream=True,
        )

        openai_client.beta.threads.messages.create(
            thread_id=thread.id, role="user", content="What is my favority fruit?"
        )

        stream_2 = openai_client.beta.threads.runs.create(
            thread_id=thread.id,
            temperature=0,
            model="gpt-3.5-turbo",
            assistant_id="any",
            stream=True,
        )

        events_2: List[AssistantStreamEvent] = []
        for event in stream_2:
            events_2.append(event)

        followup_response = assistant_stream_events_to_str_response(events_2)

        assert "banana" in followup_response


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
