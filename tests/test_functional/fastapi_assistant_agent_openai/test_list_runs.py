import logging
import pytest
from openai import OpenAI
from fastapi.testclient import TestClient

from assistant_server_openai import app


test_api = TestClient(app)


logging.getLogger("openai").setLevel(logging.DEBUG)


@pytest.fixture(scope="session")
def openai_client():
    return OpenAI(
        base_url="http://testserver/my-assistant/openai/v1",
        http_client=test_api,
    )


class TestListRuns:
    def test_list_threads(self, openai_client: OpenAI):
        thread = openai_client.beta.threads.create()

        run1 = openai_client.beta.threads.runs.create(
            assistant_id="assistant1",
            thread_id=thread.id,
            model="any",
            stream=False,
        )

        run2 = openai_client.beta.threads.runs.create(
            assistant_id="assistant1",
            thread_id=thread.id,
            model="any",
            stream=False,
        )

        thread_runs = openai_client.beta.threads.runs.list(thread_id=thread.id).data

        assert len(thread_runs) == 2
        assert run1.id in [run.id for run in thread_runs]
        assert run2.id in [run.id for run in thread_runs]
