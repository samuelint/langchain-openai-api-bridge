from typing import AsyncIterator
from langchain_core.runnables.schema import StreamEvent
from openai.types.beta import AssistantStreamEvent
from openai.types.beta.assistant_stream_event import (
    ThreadRunCreated,
    ThreadRunCompleted,
    ThreadMessageDelta,
    MessageDeltaEvent,
)
from openai.types.beta.threads import Run, MessageDelta, TextDeltaBlock, TextDelta
from langchain_openai_api_bridge.assistant.create_thread_runs_api_dto import (
    ThreadRunsDto,
)
from langchain_openai_api_bridge.assistant.repository.assistant_run_repository import (
    AssistantRunRepository,
)


class LanggraphEventToOpenAIAssistantEventStream:
    def __init__(self, run_repository: AssistantRunRepository) -> None:
        self.run_repository = run_repository

    async def to_openai_assistant_event_stream(
        self,
        astream_events: AsyncIterator[StreamEvent],
        dto: ThreadRunsDto,
    ) -> AsyncIterator[AssistantStreamEvent]:

        run = self.run_repository.create(
            assistant_id=dto.assistant_id,
            thread_id=dto.thread_id,
            model=dto.model,
            status="in_progress",
        )

        yield ThreadRunCreated(
            event="thread.run.created",
            data=run,
        )

        async for event in astream_events:
            kind = event["event"]
            match kind:
                case "on_chat_model_start":
                    print(event)
                    # yield ThreadMessageCreated(
                    #     event="thread.message.created",
                    #     data=create_message(
                    #         id="a",
                    #         thread_id="a",
                    #         role="assistant",
                    #         content="",
                    #         status="in_progress",
                    #     ),
                    # )

                case "on_chat_model_stream":
                    print(event)
                    yield self.__create_thread_message_delta(event)

                case "on_chat_model_stop":
                    print(event)

                    # yield ThreadMessageCompleted(
                    #     event="thread.message.created",
                    #     data=create_message(
                    #         id="a",
                    #         thread_id="a",
                    #         role="assistant",
                    #         content="",
                    #         status="completed",
                    #     ),
                    # )

                case "on_llm_start":
                    print(event)
                case "on_llm_stream":
                    print(event)
                case "on_llm_end":
                    print(event)
                case "on_chain_start":
                    print(event)
                case "on_chain_stream":
                    print(event)
                case "on_chain_end":
                    print(event)
                case "on_tool_start":
                    print(event)
                case "on_tool_end":
                    print(event)
                case "on_retriever_start":
                    print(event)
                case "on_retriever_end":
                    print(event)

                case "on_prompt_start":
                    print(event)
                case "on_prompt_end":
                    print(event)

        yield self.__create_thread_run_completed(run=run)

    def __create_thread_run_completed(self, run: Run):
        completed_run = run.copy()
        completed_run.status = "completed"
        completed_run = self.run_repository.update(completed_run)

        return ThreadRunCompleted(
            event="thread.run.completed",
            data=completed_run,
        )

    def __create_thread_message_delta(self, event: StreamEvent):
        content = event["data"]["chunk"].content

        return ThreadMessageDelta(
            event="thread.message.delta",
            data=MessageDeltaEvent(
                id="id-1",
                delta=MessageDelta(
                    content=[
                        TextDeltaBlock(
                            index=0,
                            type="text",
                            text=TextDelta(value=content),
                        )
                    ],
                    role="assistant",
                ),
                object="thread.message.delta",
            ),
        )
