import logging
import pytest
from openai import OpenAI

from fastapi.testclient import TestClient
from assistant_server_openai import app
from tests.test_functional.assets.assets import base64_url_pig_image
from tests.test_functional.assistant_stream_utils import (
    assistant_stream_events_to_str_response,
)


test_api = TestClient(app)


logging.getLogger("openai").setLevel(logging.DEBUG)


@pytest.fixture(scope="session")
def openai_client():
    return OpenAI(
        base_url="http://testserver/my-assistant/openai/v1",
        http_client=test_api,
    )


class TestMultiModal:

    @pytest.fixture
    def base64_pig_image(self):
        return base64_url_pig_image()

    def test_run_stream_message_deltas(
        self, openai_client: OpenAI, base64_pig_image: str
    ):
        thread = openai_client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is in the image?",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": base64_pig_image},
                        },
                    ],
                },
            ]
        )

        stream = openai_client.beta.threads.runs.create(
            thread_id=thread.id,
            model="gpt-4o",
            assistant_id="any",
            stream=True,
            temperature=0,
        )

        str_response = assistant_stream_events_to_str_response(stream)

        assert "pig" in str_response.lower()
