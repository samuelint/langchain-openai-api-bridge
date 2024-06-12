import pytest
from langchain_core.runnables.schema import StreamEvent
from langchain_openai_api_bridge.assistant.adapter.on_tool_start_handler import (
    OnToolStartHandler,
)
from langchain_openai_api_bridge.assistant.create_thread_runs_api_dto import (
    ThreadRunsDto,
)


@pytest.fixture
def some_thread_dto():
    return ThreadRunsDto(
        assistant_id="assistant1",
        thread_id="thread1",
        model="some-model",
    )


class TestOnToolStartHandler:

    @pytest.fixture
    def instance(self):
        return OnToolStartHandler()

    def test_input_arguments(
        self, instance: OnToolStartHandler, some_thread_dto: ThreadRunsDto
    ):
        event = StreamEvent(
            run_id="r1",
            event="on_tool_start",
            name="my_tool",
            data={"input": {"say": "hello"}},
        )

        result = instance.handle(event=event, dto=some_thread_dto)

        assert (
            result[0].data.step_details.tool_calls[0].function.arguments
            == '{"say": "hello"}'
        )

    def test_no_input_raise_exception(
        self, instance: OnToolStartHandler, some_thread_dto: ThreadRunsDto
    ):
        event = StreamEvent(
            run_id="r1",
            event="on_tool_start",
            name="my_tool",
            data={},
        )

        with pytest.raises(Exception):
            instance.handle(event=event, dto=some_thread_dto)
