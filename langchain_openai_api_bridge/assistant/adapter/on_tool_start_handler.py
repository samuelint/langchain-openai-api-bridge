from langchain_core.runnables.schema import StreamEvent
from openai.types.beta import AssistantStreamEvent

from langchain_openai_api_bridge.assistant.adapter.openai_event_factory import (
    create_langchain_tool_run_step_tools_created,
)
from langchain_openai_api_bridge.assistant.create_thread_runs_api_dto import (
    ThreadRunsDto,
)


class OnToolStartHandler:
    def __init__(self) -> None:
        pass

    def handle(
        self, event: StreamEvent, dto: ThreadRunsDto
    ) -> list[AssistantStreamEvent]:
        step_id = event["run_id"]
        name = event["name"]
        arguments = event["data"]["input"]
        metadata = event.get("metadata", None)
        tool_created_event = create_langchain_tool_run_step_tools_created(
            step_id=step_id,
            assistant_id=dto.assistant_id,
            thread_id=dto.thread_id,
            status="in_progress",
            name=name,
            arguments=arguments,
            metadata=metadata,
        )

        return [tool_created_event]
