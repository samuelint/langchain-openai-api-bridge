import pytest
from openai import OpenAI

from fastapi.testclient import TestClient
from assistant_server_groq import app
from tests.test_functional.assistant_stream_utils import (
    assistant_stream_events_to_str_response,
)


test_api = TestClient(app)


@pytest.fixture(scope="session")
def openai_client():
    return OpenAI(
        base_url="http://testserver/my-groq-assistant/openai/v1",
        http_client=test_api,
    )


class TestGroqAssistant:

    def test_run_stream_message_deltas(
        self,
        openai_client: OpenAI,
    ):
        thread = openai_client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": "Hello!",
                },
            ]
        )

        stream = openai_client.beta.threads.runs.create(
            thread_id=thread.id,
            model="llama3-8b-8192",
            assistant_id="any",
            stream=True,
            temperature=0,
        )

        str_response = assistant_stream_events_to_str_response(stream)

        assert len(str_response) > 0
