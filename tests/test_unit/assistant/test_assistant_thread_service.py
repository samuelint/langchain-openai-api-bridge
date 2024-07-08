import pytest
from langchain_openai_api_bridge.assistant.assistant_thread_service import (
    AssistantThreadService,
)
from langchain_openai_api_bridge.assistant.create_thread_api_dto import CreateThreadDto
from langchain_openai_api_bridge.assistant.repository import (
    InMemoryMessageRepository,
    InMemoryRunRepository,
    InMemoryThreadRepository,
)


@pytest.fixture
def run_repository() -> InMemoryRunRepository:
    return InMemoryRunRepository()


@pytest.fixture
def message_repository() -> InMemoryMessageRepository:
    return InMemoryMessageRepository()


@pytest.fixture
def instance(run_repository, message_repository) -> AssistantThreadService:
    instance = AssistantThreadService(
        thread_repository=InMemoryThreadRepository(),
        message_repository=message_repository,
        run_repository=run_repository,
    )

    return instance


class TestCreate:

    def test_create(self, instance: AssistantThreadService):
        thread = instance.create(dto=CreateThreadDto(messages=[]))

        retreived_thread = instance.retreive(thread_id=thread.id)

        assert retreived_thread is not None


class TestDelete:

    def test_delete_thread(self, instance: AssistantThreadService):
        thread = instance.create(dto=CreateThreadDto(messages=[]))

        instance.delete(thread_id=thread.id)

        retreived_thread = instance.retreive(thread_id=thread.id)
        assert retreived_thread is None

    def test_runs_associated_with_thread_are_deleted(
        self, instance: AssistantThreadService, run_repository: InMemoryRunRepository
    ):
        thread = instance.create(dto=CreateThreadDto(messages=[]))
        run = run_repository.create(
            assistant_id="assistant1",
            thread_id=thread.id,
            model="any",
            status="in_progress",
        )

        instance.delete(thread_id=thread.id)

        retreive_run = run_repository.retreive(run_id=run.id)

        assert retreive_run is None

    def test_messages_associated_with_thread_are_deleted(
        self,
        instance: AssistantThreadService,
        message_repository: InMemoryMessageRepository,
    ):

        thread = instance.create(dto=CreateThreadDto(messages=[]))
        messages = message_repository.create_many(
            thread_id=thread.id,
            messages=[
                {"content": "hello", "role": "user"},
                {"content": "hi human", "role": "assistant"},
            ],
        )

        instance.delete(thread_id=thread.id)

        for message in messages:
            retreived_message = message_repository.retreive(
                message_id=message.id, thread_id=thread.id
            )
            assert retreived_message is None
