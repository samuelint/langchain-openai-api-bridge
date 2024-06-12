from decoy import Decoy
import pytest

from langchain_openai_api_bridge.assistant.adapter.on_chat_model_stream_handler import (
    OnChatModelStreamHandler,
)
from langchain_openai_api_bridge.assistant.create_thread_runs_api_dto import (
    ThreadRunsDto,
)
from langchain_openai_api_bridge.assistant.openai_message_factory import create_message
from langchain_openai_api_bridge.assistant.repository.message_repository import (
    MessageRepository,
)
from tests.test_unit.core.agent_stream_utils import create_stream_chunk_event


@pytest.fixture
def thread_message_repository(decoy: Decoy):
    return decoy.mock(cls=MessageRepository)


@pytest.fixture
def some_thread_dto():
    return ThreadRunsDto(
        assistant_id="assistant1",
        thread_id="thread1",
        model="some-model",
    )


@pytest.fixture
def instance(
    thread_message_repository: MessageRepository,
):
    return OnChatModelStreamHandler(
        thread_message_repository=thread_message_repository,
    )


class TestOnChatModelStreamHandler:

    def test_when_message_exist_delta_is_returned(
        self,
        decoy: Decoy,
        thread_message_repository: MessageRepository,
        instance: OnChatModelStreamHandler,
        some_thread_dto: ThreadRunsDto,
    ):
        event = create_stream_chunk_event(
            run_id="a", event="on_chat_model_stream", content=" World!"
        )
        decoy.when(
            thread_message_repository.retreive_message_id_by_run_id(
                run_id="a", thread_id="thread1"
            )
        ).then_return("1")

        result = instance.handle(event=event, dto=some_thread_dto)

        assert result[0].event == "thread.message.delta"
        assert result[0].data.delta.content[0].text.value == " World!"

    def test_not_persisted_message_is_created_and_then_content_chunk_is_put_as_delta_event(
        self,
        decoy: Decoy,
        thread_message_repository: MessageRepository,
        instance: OnChatModelStreamHandler,
        some_thread_dto: ThreadRunsDto,
    ):
        event = create_stream_chunk_event(
            run_id="a", event="on_chat_model_stream", content="Hello"
        )

        decoy.when(
            thread_message_repository.retreive_message_id_by_run_id(
                run_id="a", thread_id="thread1"
            )
        ).then_return(None)
        created_message = create_message(
            id="1", thread_id="thread1", role="assistant", content=""
        )
        decoy.when(
            thread_message_repository.create(
                thread_id="thread1",
                role="assistant",
                content="",
                status="in_progress",
                run_id="a",
            )
        ).then_return(created_message)

        result = instance.handle(event=event, dto=some_thread_dto)

        assert result[0].event == "thread.message.created"
        assert result[0].data == created_message
        assert result[1].event == "thread.message.delta"
        assert result[1].data.delta.content[0].text.value == "Hello"

    def test_event_without_content_returns_no_events(
        self, instance: OnChatModelStreamHandler, some_thread_dto: ThreadRunsDto
    ):
        event = create_stream_chunk_event(
            run_id="a", event="on_chat_model_stream", content=""
        )

        result = instance.handle(event=event, dto=some_thread_dto)

        assert len(result) == 0
