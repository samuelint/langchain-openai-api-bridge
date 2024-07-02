from typing import List
import pytest
from openai import OpenAI
from openai.types.beta import AssistantStreamEvent, Thread

from fastapi.testclient import TestClient
from with_injector_assistant_server_openai import app
from tests.test_functional.assistant_stream_utils import (
    assistant_stream_events_to_str_response,
)


test_api = TestClient(app)


@pytest.fixture(scope="session")
def openai_client():
    return OpenAI(
        base_url="http://testserver/my-assistant/openai/v1",
        http_client=test_api,
    )


class TestFollowupMessage:

    @pytest.fixture(scope="session")
    def thread(self, openai_client: OpenAI) -> Thread:
        return openai_client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": "Remember that my favorite fruit is banana. I Like bananas.",
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
