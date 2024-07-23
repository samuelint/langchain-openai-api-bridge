import pytest
from openai import OpenAI

from fastapi.testclient import TestClient
from assistant_server_llamacpp import app
from tests.test_functional.assistant_stream_utils import (
    assistant_stream_events_to_str_response,
)


test_api = TestClient(app)


@pytest.fixture(scope="session")
def openai_client():
    return OpenAI(
        base_url="http://testserver/my-llamacpp-assistant/openai/v1",
        http_client=test_api,
    )


class TestLlamaCppAssistant:

    def test_run_stream_response_has_no_undesired_characters(
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
            model="llama3",
            assistant_id="any",
            stream=True,
            temperature=0,
        )

        str_response = assistant_stream_events_to_str_response(stream)

        assert len(str_response) > 0
        assert "&#39;" not in str_response

    # def test_function_calling(
    #     self,
    #     openai_client: OpenAI,
    # ):
    #     thread = openai_client.beta.threads.create(
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": "What is the magic number of 125?",
    #             },
    #         ]
    #     )

    #     stream = openai_client.beta.threads.runs.create(
    #         thread_id=thread.id,
    #         model="llama3",
    #         assistant_id="any",
    #         stream=True,
    #         temperature=0,
    #     )

    #     str_response = assistant_stream_events_to_str_response(stream)

    #     assert "127" in str_response


if __name__ == "__main__":
    pytest.main([__file__])
