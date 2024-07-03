import logging
import pytest
from openai import OpenAI
from openai.types.beta import Thread
from fastapi.testclient import TestClient
import validators
from assistant_server_openai import app
from openai.pagination import SyncCursorPage
from openai._base_client import (
    make_request_options,
)
from openai._utils import (
    maybe_transform,
)
from openai.types.beta import (
    thread_update_params,
)

test_api = TestClient(app)


logging.getLogger("openai").setLevel(logging.DEBUG)


@pytest.fixture(scope="session")
def openai_client():
    return OpenAI(
        base_url="http://testserver/my-assistant/openai/v1",
        http_client=test_api,
    )


class TestThread:
    def test_create_empty_thread(self, openai_client: OpenAI):
        thread = openai_client.beta.threads.create()

        assert validators.uuid(thread.id)

    def test_create_update_thread(self, openai_client: OpenAI):
        thread = openai_client.beta.threads.create(metadata={"key": "value"})

        openai_client.beta.threads.update(
            thread_id=thread.id, metadata={"key": "value2"}
        )
        retreived_thread = openai_client.beta.threads.retrieve(thread_id=thread.id)

        assert retreived_thread.metadata == {"key": "value2"}

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

    def test_list_threads(self, openai_client: OpenAI):
        created_thread = openai_client.beta.threads.create(metadata={"key": "value"})

        page = openai_client.beta.threads._get_api_list(
            "/threads",
            page=SyncCursorPage[Thread],
            options=make_request_options(
                # extra_headers=extra_headers,
                # extra_query=extra_query,
                # extra_body=extra_body,
                # timeout=timeout,
                query=maybe_transform(
                    {
                        # "after": after,
                        # "before": before,
                        # "limit": limit,
                        # "order": order,
                    },
                    thread_update_params.ThreadUpdateParams,
                ),
            ),
            model=Thread,
        )
        threads = page.data

        assert len(threads) > 0
        assert created_thread.id in [thread.id for thread in threads]
