import logging
import pytest
from openai import OpenAI
from fastapi.testclient import TestClient

from assistant_server_custom_service import app


test_api = TestClient(app)


logging.getLogger("openai").setLevel(logging.DEBUG)


@pytest.fixture(scope="session")
def openai_client():
    return OpenAI(
        base_url="http://testserver/my-custom-assistant/openai/v1",
        http_client=test_api,
    )


class TestThreadWithCustomMetadata:
    def test_thread_contains_custom_metadata(self, openai_client: OpenAI):
        thread = openai_client.beta.threads.create()

        assert thread.metadata["custom_metadata"] == "my_custom_metadata"
