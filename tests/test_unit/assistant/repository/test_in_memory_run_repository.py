import pytest
import validators

from langchain_openai_api_bridge.assistant.repository.in_memory_run_repository import (
    InMemoryRunRepository,
)


class TestCreateInMemoryRun:

    @pytest.fixture
    def instance(self):
        return InMemoryRunRepository()

    def test_created_contains_uuid_id(self, instance: InMemoryRunRepository):
        result = instance.create(
            assistant_id="A", thread_id="T1", model="some", status="in_progress"
        )

        assert validators.uuid(result.id)

    def test_created_run_is_persisted_with_assistant_id(
        self, instance: InMemoryRunRepository
    ):
        created_run = instance.create(
            assistant_id="A", thread_id="T1", model="some", status="in_progress"
        )

        result = instance.retreive(created_run.id)

        assert result.assistant_id == "A"

    def test_created_run_is_persisted_with_thread_id(
        self, instance: InMemoryRunRepository
    ):
        created_run = instance.create(
            assistant_id="A", thread_id="T1", model="some", status="in_progress"
        )

        result = instance.retreive(created_run.id)

        assert result.thread_id == "T1"

    def test_created_run_is_persisted_with_model(self, instance: InMemoryRunRepository):
        created_run = instance.create(
            assistant_id="A", thread_id="T1", model="some", status="in_progress"
        )

        result = instance.retreive(created_run.id)

        assert result.model == "some"

    def test_created_run_is_persisted_with_status(
        self, instance: InMemoryRunRepository
    ):
        created_run = instance.create(
            assistant_id="A", thread_id="T1", model="some", status="in_progress"
        )

        result = instance.retreive(created_run.id)

        assert result.status == "in_progress"

    def test_created_run_is_persisted_with_instrctions(
        self, instance: InMemoryRunRepository
    ):
        created_run = instance.create(
            assistant_id="A",
            thread_id="T1",
            model="some",
            status="in_progress",
            instructions="some instructions",
        )

        result = instance.retreive(created_run.id)

        assert result.instructions == "some instructions"

    def test_retreived_run_is_immuable(self, instance: InMemoryRunRepository):
        created_run = instance.create(
            assistant_id="A",
            thread_id="T1",
            model="some",
            status="in_progress",
            instructions="some instructions",
        )
        created_run.model = "new-model"
        result = instance.retreive(run_id=created_run.id)

        assert result.model == "some"


class TestUpdateInMemoryRun:

    @pytest.fixture
    def instance(self):
        return InMemoryRunRepository()

    def test_run_is_updated(self, instance: InMemoryRunRepository):
        created = instance.create(
            assistant_id="A", thread_id="T1", model="some", status="in_progress"
        )

        created.model = "new-model"
        updated = instance.update(run=created)
        retreived = instance.retreive(run_id=updated.id)

        assert retreived.model == "new-model"
        assert updated.model == "new-model"


class TestDeleteInMemoryRun:

    @pytest.fixture
    def instance(self):
        return InMemoryRunRepository()

    def test_run_is_deleted_using_dto(self, instance: InMemoryRunRepository):
        created = instance.create(
            assistant_id="A", thread_id="T1", model="some", status="in_progress"
        )

        instance.delete(run=created)

        retreived = instance.retreive(run_id=created.id)
        assert retreived is None

    def test_run_is_deleted_using_id(self, instance: InMemoryRunRepository):
        created = instance.create(
            assistant_id="A", thread_id="T1", model="some", status="in_progress"
        )

        instance.delete(run_id=created.id)

        retreived = instance.retreive(run_id=created.id)
        assert retreived is None
