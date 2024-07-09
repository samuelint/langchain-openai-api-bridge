from decoy import Decoy, matchers
import pytest

from langchain_openai_api_bridge.assistant.adapter.on_chat_model_end_handler import (
    OnChatModelEndHandler,
)

from langchain_openai_api_bridge.assistant.create_thread_runs_api_dto import (
    ThreadRunsDto,
)
from langchain_openai_api_bridge.assistant.adapter.openai_message_factory import (
    create_message,
)
from langchain_openai_api_bridge.assistant.repository.message_repository import (
    MessageRepository,
)
from tests.test_unit.core.agent_stream_utils import create_stream_output_event
from openai.types.beta.threads import Message
from openai.types.beta.threads import Run


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
def some_message() -> Message:
    return create_message(
        id="1", thread_id="thread1", role="assistant", content="Hello"
    )


@pytest.fixture
def some_run() -> Run:
    return Run(
        id="a",
        assistant_id="assistant1",
        created_at=0,
        tools=[],
        thread_id="thread1",
        model="some-model",
        status="in_progress",
        instructions="",
        object="thread.run",
        parallel_tool_calls=True,
    )


@pytest.fixture
def instance(
    thread_message_repository: MessageRepository,
):
    return OnChatModelEndHandler(
        thread_message_repository=thread_message_repository,
    )


class TestOnChatModelStreamHandler:

    def test_message_not_existing_in_database_returns_no_events(
        self,
        decoy: Decoy,
        thread_message_repository: MessageRepository,
        instance: OnChatModelEndHandler,
        some_thread_dto: ThreadRunsDto,
        some_run: Run,
    ):
        event = create_stream_output_event(run_id="a", event="on_chat_model_end")
        decoy.when(
            thread_message_repository.retreive_unique_by_run_id(
                run_id="a", thread_id=some_thread_dto.thread_id
            )
        ).then_return(None)

        result = instance.handle(event=event, dto=some_thread_dto, run=some_run)

        assert len(result) == 0

    def test_message_existing_in_database_is_completed_with_final_content(
        self,
        decoy: Decoy,
        thread_message_repository: MessageRepository,
        instance: OnChatModelEndHandler,
        some_thread_dto: ThreadRunsDto,
        some_message: Message,
        some_run: Run,
    ):
        event = create_stream_output_event(
            run_id="a", event="on_chat_model_end", content="hello world!"
        )
        decoy.when(
            thread_message_repository.retreive_unique_by_run_id(
                run_id="a", thread_id=some_thread_dto.thread_id
            )
        ).then_return(some_message)
        decoy.when(thread_message_repository.update(matchers.Anything())).then_do(
            lambda message: message
        )

        result = instance.handle(event=event, dto=some_thread_dto, run=some_run)

        assert result[0].event == "thread.message.completed"
        assert result[0].data.content[0].text.value == "hello world!"

    def test_message_existing_in_database_is_completed_with_completed_status(
        self,
        decoy: Decoy,
        thread_message_repository: MessageRepository,
        instance: OnChatModelEndHandler,
        some_thread_dto: ThreadRunsDto,
        some_message: Message,
        some_run: Run,
    ):
        event = create_stream_output_event(
            run_id="a", event="on_chat_model_end", content="hello world!"
        )
        decoy.when(
            thread_message_repository.retreive_unique_by_run_id(
                run_id="a", thread_id=some_thread_dto.thread_id
            )
        ).then_return(some_message)
        decoy.when(thread_message_repository.update(matchers.Anything())).then_do(
            lambda message: message
        )

        result = instance.handle(event=event, dto=some_thread_dto, run=some_run)

        assert result[0].data.status == "completed"
        decoy.verify(
            thread_message_repository.update(
                matchers.HasAttributes({"id": "1", "status": "completed"})
            )
        )
