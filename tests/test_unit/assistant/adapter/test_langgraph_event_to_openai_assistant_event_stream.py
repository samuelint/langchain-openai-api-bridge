from typing import List
import pytest
from unittest.mock import Mock

import validators
from langchain_openai_api_bridge.assistant.adapter.langgraph_event_to_openai_assistant_event_stream import (
    LanggraphEventToOpenAIAssistantEventStream,
)
from langchain_openai_api_bridge.assistant.create_thread_runs_api_dto import (
    ThreadRunsDto,
)
from langchain_openai_api_bridge.assistant.openai_run_factory import create_run
from langchain_openai_api_bridge.assistant.repository.assistant_message_repository import (
    AssistantMessageRepository,
)
from langchain_openai_api_bridge.assistant.repository.assistant_run_repository import (
    AssistantRunRepository,
)
from langchain_openai_api_bridge.assistant.repository.in_memory_message_repository import (
    InMemoryMessageRepository,
)
from tests.stream_utils import assemble_stream, generate_stream
from tests.test_unit.core.agent_stream_utils import (
    create_stream_chunk_event,
    create_stream_output_event,
)
from langchain_core.runnables.schema import StreamEvent


class TestToOpenAIAssistantEventStream:
    dto = ThreadRunsDto(
        thread_id="thread1",
        assistant_id="assistant1",
        model="some-model",
    )
    created_run = create_run(
        id="run1",
        assistant_id=dto.assistant_id,
        thread_id=dto.thread_id,
        model=dto.model,
        status="in_progress",
    )

    @pytest.fixture
    def run_repository_mock(self):
        mock = Mock(spec=AssistantRunRepository)

        mock.create.return_value = self.created_run
        mock.update.side_effect = lambda run: run

        return mock

    @pytest.fixture
    def thread_message_repository_mock(self):
        repo = InMemoryMessageRepository()

        return repo

    @pytest.fixture
    def instance(
        self,
        run_repository_mock: AssistantRunRepository,
        thread_message_repository_mock: AssistantMessageRepository,
    ):
        return LanggraphEventToOpenAIAssistantEventStream(
            run_repository=run_repository_mock,
            thread_message_repository=thread_message_repository_mock,
        )

    async def __execute_stream_with_mock(
        self,
        instance: LanggraphEventToOpenAIAssistantEventStream,
        stream_events: List[StreamEvent] = [],
    ) -> list[object]:

        astream_events = generate_stream(stream_events)

        response_stream = instance.to_openai_assistant_event_stream(
            astream_events=astream_events, dto=self.dto
        )

        return await assemble_stream(response_stream)

    @pytest.mark.asyncio
    async def test_run_is_created_on_stream_start(
        self,
        instance: LanggraphEventToOpenAIAssistantEventStream,
    ):
        items = await self.__execute_stream_with_mock(instance=instance)

        assert items[0].event == "thread.run.created"
        assert items[0].data.id == "run1"
        assert items[0].data.thread_id == "thread1"
        assert items[0].data.assistant_id == "assistant1"
        assert items[0].data.model == "some-model"
        assert items[0].data.status == "in_progress"

    @pytest.mark.asyncio
    async def test_last_event_is_thread_run_completed(
        self,
        instance: LanggraphEventToOpenAIAssistantEventStream,
    ):
        items = await self.__execute_stream_with_mock(instance=instance)

        assert items[-1].event == "thread.run.completed"

    @pytest.mark.asyncio
    async def test_thread_run_completed_status_is_completed(
        self,
        instance: LanggraphEventToOpenAIAssistantEventStream,
    ):
        items = await self.__execute_stream_with_mock(instance=instance)

        assert items[-1].data.status == "completed"

    @pytest.mark.asyncio
    async def test_thread_run_completed_status_is_persisted(
        self,
        instance: LanggraphEventToOpenAIAssistantEventStream,
    ):
        await self.__execute_stream_with_mock(instance=instance)

        assert instance.run_repository.update.called
        call_args = instance.run_repository.update.call_args
        run_arg = call_args[0][0]
        assert run_arg.status == "completed"

    @pytest.mark.asyncio
    async def test_message_is_created_on_chat_model_start(
        self,
        instance: LanggraphEventToOpenAIAssistantEventStream,
    ):
        items = await self.__execute_stream_with_mock(
            instance=instance,
            stream_events=[
                create_stream_chunk_event(run_id="a", event="on_chat_model_start"),
                create_stream_chunk_event(
                    run_id="a", event="on_chat_model_stream", content="hello"
                ),
                create_stream_output_event(run_id="a", event="on_chat_model_end"),
            ],
        )

        assert items[1].event == "thread.message.created"
        assert validators.uuid(items[1].data.id)

    @pytest.mark.asyncio
    async def test_run_stream_contains_message_delta(
        self,
        instance: LanggraphEventToOpenAIAssistantEventStream,
    ):
        items = await self.__execute_stream_with_mock(
            instance=instance,
            stream_events=[
                create_stream_chunk_event(run_id="a", event="on_chat_model_start"),
                create_stream_chunk_event(
                    run_id="a", event="on_chat_model_stream", content="some"
                ),
                create_stream_chunk_event(
                    run_id="a", event="on_chat_model_stream", content=" content!"
                ),
                create_stream_output_event(run_id="a", event="on_chat_model_end"),
            ],
        )

        assert items[2].event == "thread.message.delta"
        assert items[2].data.id == items[1].data.id
        assert items[2].data.delta.content[0].text.value == "some"
        assert items[3].event == "thread.message.delta"
        assert items[3].data.id == items[2].data.id
        assert items[3].data.delta.content[0].text.value == " content!"

    @pytest.mark.asyncio
    async def test_message_completed_event_is_sent(
        self,
        instance: LanggraphEventToOpenAIAssistantEventStream,
    ):
        items = await self.__execute_stream_with_mock(
            instance=instance,
            stream_events=[
                create_stream_chunk_event(run_id="a", event="on_chat_model_start"),
                create_stream_chunk_event(
                    run_id="a", event="on_chat_model_stream", content="hello"
                ),
                create_stream_chunk_event(
                    run_id="a", event="on_chat_model_stream", content=" world!"
                ),
                create_stream_output_event(
                    run_id="a", event="on_chat_model_end", content="hello world!"
                ),
            ],
        )

        assert items[4].event == "thread.message.completed"
        assert items[4].data.id == items[1].data.id
        assert items[4].data.status == "completed"
        assert items[4].data.content.text.value == "hello world!"
