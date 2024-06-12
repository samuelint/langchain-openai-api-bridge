from langchain_core.runnables.schema import StreamEvent
from openai.types.beta import AssistantStreamEvent

from langchain_openai_api_bridge.assistant.adapter.openai_event_factory import (
    create_langchain_tool_thread_run_step_completed,
)
from langchain_openai_api_bridge.assistant.create_thread_runs_api_dto import (
    ThreadRunsDto,
)


class OnToolEndHandler:
    def __init__(self) -> None:
        pass

    def handle(
        self, event: StreamEvent, dto: ThreadRunsDto
    ) -> list[AssistantStreamEvent]:
        step_id = event["run_id"]
        name = event["name"]
        arguments = event["data"].get("input", None)
        output = event["data"].get("output", None)
        metadata = event["metadata"]
        tool_completed_event = create_langchain_tool_thread_run_step_completed(
            step_id=step_id,
            assistant_id=dto.assistant_id,
            thread_id=dto.thread_id,
            status="in_progress",
            name=name,
            arguments=arguments,
            output=output,
            metadata=metadata,
        )

        return [tool_completed_event]
