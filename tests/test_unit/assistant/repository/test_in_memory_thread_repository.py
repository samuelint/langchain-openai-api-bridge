from pydantic import BaseModel
import validators
from unittest.mock import patch
from langchain_openai_api_bridge.assistant.repository.in_memory_thread_repository import (
    InMemoryThreadRepository,
)


class SomeMetadata(BaseModel):
    a: str = None


class TestInMemoryThreadRepository:
    instance = InMemoryThreadRepository()

    def test_created_thread_contains_uuid_id(self):
        result = self.instance.create()

        assert validators.uuid(result.id)

    def test_created_thread_contains_metadata(self):
        metadata = SomeMetadata(a="AAA")

        result = self.instance.create(metadata=metadata)

        assert result.metadata.a == "AAA"

    @patch("time.time", return_value=1638316800)
    def test_created_thread_contains_created_at(self, mock_time):
        result = self.instance.create()

        assert result.created_at == 1638316800

    def test_created_thread_is_retreivable(self):
        created = self.instance.create()
        retreived = self.instance.retreive(created.id)

        assert retreived.id == created.id

    def test_retreived_thread_is_immuable(self):
        metadata = SomeMetadata(a="AAA")
        created = self.instance.create(metadata=metadata)
        created.metadata.a = "B"

        retreived = self.instance.retreive(created.id)

        assert retreived.metadata.a == "AAA"

    def test_thread_is_deleted(self):
        created = self.instance.create()

        self.instance.delete(created.id)
        retreived = self.instance.retreive(created.id)

        assert retreived is None

    def test_deleted_tread_return_value_contains_id(self):
        created = self.instance.create()

        result = self.instance.delete(created.id)

        assert result.id == created.id
