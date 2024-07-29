import logging
import pytest
from openai import OpenAI

from fastapi.testclient import TestClient
from assistant_server_openai import app
from tests.test_functional.assistant_stream_utils import (
    assistant_stream_events,
    assistant_stream_events_to_str_response,
    validate_llm_response,
)
from openai.types.beta import AssistantStreamEvent

test_api = TestClient(app)


logging.getLogger("openai").setLevel(logging.DEBUG)


@pytest.fixture(scope="session")
def openai_client():
    return OpenAI(
        base_url="http://testserver/my-assistant/openai/v1",
        http_client=test_api,
    )


class TestSimpleToolCalling:

    def test_simple_tool_is_called(self, openai_client: OpenAI):
        thread = openai_client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": "What is the magic number of 45?",
                },
            ]
        )
        stream = openai_client.beta.threads.runs.create(
            thread_id=thread.id,
            model="gpt-4o-mini",
            assistant_id="any",
            temperature=0,
            stream=True,
        )

        events = assistant_stream_events(stream)
        str_response = assistant_stream_events_to_str_response(events)

        assert "47" in str_response


class TestSubAgentToolCalling:

    @pytest.fixture(scope="session")
    def result_events(self, openai_client: OpenAI):
        thread = openai_client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": "Tell a joke about cats.",
                },
            ]
        )
        stream = openai_client.beta.threads.runs.create(
            thread_id=thread.id,
            model="gpt-4o-mini",
            assistant_id="any",
            temperature=0,
            stream=True,
        )

        return assistant_stream_events(stream)

    def test_the_same_answer_is_not_repeated(
        self, result_events: list[AssistantStreamEvent]
    ):
        str_response = assistant_stream_events_to_str_response(result_events)

        is_a_repetition = validate_llm_response(
            question="Is the same message a repeted?",
            str_response=str_response,
        )

        assert "no" in is_a_repetition

    def test_a_run_step_is_defined_for_tool_usage(
        self, result_events: list[AssistantStreamEvent]
    ):
        events_iter = iter(result_events)
        found_thread_run_step_created = False
        found_thread_run_step_completed = False
        for event in events_iter:
            if event.event == "thread.run.step.created":
                found_thread_run_step_created = True
            elif event.event == "thread.run.step.completed":
                found_thread_run_step_completed = True
                break
        else:
            assert False, "Expected thread.run.step.completed in result_events"

        assert found_thread_run_step_created
        assert found_thread_run_step_completed
