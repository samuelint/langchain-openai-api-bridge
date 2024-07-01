import pytest
import validators

from langchain_openai_api_bridge.assistant.repository.in_memory_message_repository import (
    InMemoryMessageRepository,
)
from openai.types.beta.threads import TextContentBlock, Text
from openai.types.beta import thread_create_params


some_text_content_1 = TextContentBlock(
    text=Text(value="AAA", annotations=[]), type="text"
)
some_text_content_2 = TextContentBlock(
    text=Text(value="BBB", annotations=[]), type="text"
)


class TestCreateInMemoryMessageRepository:

    @pytest.fixture
    def instance(self):
        return InMemoryMessageRepository()

    def test_created_contains_uuid_id(self, instance: InMemoryMessageRepository):
        result = instance.create(
            thread_id="A",
            role="user",
            content=[some_text_content_1],
        )

        assert validators.uuid(result.id)

    def test_created_contains_role(self, instance: InMemoryMessageRepository):
        result = instance.create(
            thread_id="A",
            role="user",
            content=[some_text_content_1],
        )

        assert result.role == "user"

    def test_created_contains_thread_id(self, instance: InMemoryMessageRepository):
        result = instance.create(
            thread_id="A",
            role="user",
            content=[some_text_content_1],
        )

        assert result.thread_id == "A"

    def test_created_contains_content(self, instance: InMemoryMessageRepository):
        result = instance.create(
            thread_id="A",
            role="user",
            content=[some_text_content_1],
        )

        assert result.content[0].text.value == "AAA"

    def test_create_many_with_content(self, instance: InMemoryMessageRepository):
        result = instance.create_many(
            thread_id="A",
            messages=[
                thread_create_params.Message(
                    role="user", content=[some_text_content_1]
                ),
                thread_create_params.Message(
                    role="assistant", content=[some_text_content_2]
                ),
            ],
        )

        assert result[0].content[0].text.value == "AAA"
        assert result[1].content[0].text.value == "BBB"

    def test_create_many_with_string(self, instance: InMemoryMessageRepository):
        result = instance.create_many(
            thread_id="A",
            messages=[
                thread_create_params.Message(role="user", content="Hello"),
                thread_create_params.Message(role="assistant", content="World"),
            ],
        )

        assert result[0].content[0].text.value == "Hello"
        assert result[1].content[0].text.value == "World"

    def test_created_contains_run_id(self, instance: InMemoryMessageRepository):
        created = instance.create(
            thread_id="A",
            role="user",
            content="",
            run_id="123",
        )

        retreived = instance.retreive(message_id=created.id, thread_id="A")

        assert retreived.run_id == "123"

    def test_created_contains_metadata(self, instance: InMemoryMessageRepository):
        created = instance.create(
            thread_id="A",
            role="user",
            content="",
            metadata={"key": "value"},
        )

        retreived = instance.retreive(message_id=created.id, thread_id="A")

        assert retreived.metadata["key"] == "value"


class TestRetreiveInMemoryMessageRepository:
    @pytest.fixture
    def instance(self):
        return InMemoryMessageRepository()

    def test_created_message_is_retreivable(self, instance):
        created = instance.create(
            thread_id="A",
            role="user",
            content=[some_text_content_1],
        )
        result = instance.retreive(message_id=created.id, thread_id="A")

        assert result.id == created.id

    def test_retreived_message_is_immuable(self, instance):
        created = instance.create(
            thread_id="A",
            role="user",
            content=[some_text_content_1],
        )
        created.role = "assistant"
        result = instance.retreive(message_id=created.id, thread_id="A")

        assert result.role == "user"

    def test_not_existing_message_return_none(self, instance):
        result = instance.retreive(message_id="not-existing", thread_id="A")

        assert result is None


class TestRetreiveInMemoryMessageByRunId:
    @pytest.fixture
    def instance(self):
        return InMemoryMessageRepository(data={})

    def test_message_is_retreivable_by_run_id(
        self, instance: InMemoryMessageRepository
    ):
        expected = instance.create(
            thread_id="A",
            role="user",
            content=[some_text_content_1],
            run_id="123",
        )
        instance.create(
            thread_id="A",
            role="user",
            content=[some_text_content_2],
            run_id="1234",
        )
        instance.create(
            thread_id="B",
            role="user",
            content=[some_text_content_1],
            run_id="123",
        )
        result = instance.retreive_unique_by_run_id(run_id="123", thread_id="A")

        assert result.id == expected.id
        assert result.content == expected.content

    def test_not_found_return_none(self, instance: InMemoryMessageRepository):

        result = instance.retreive_unique_by_run_id(run_id="123", thread_id="Z")

        assert result is None


class TestUpdateInMemoryMessage:
    @pytest.fixture
    def instance(self):
        return InMemoryMessageRepository()

    def test_update_content(self, instance: InMemoryMessageRepository):
        created = instance.create(
            thread_id="A",
            role="user",
            content=[some_text_content_1],
        )
        created.content = [some_text_content_2]
        instance.update(created)
        result = instance.retreive(message_id=created.id, thread_id="A")

        assert result.content == [some_text_content_2]


class TestDeleteInMemoryMessageRepository:
    @pytest.fixture
    def instance(self):
        return InMemoryMessageRepository()

    def test_deleted_message_doesnt_exist(self, instance):
        created = instance.create(
            thread_id="A",
            role="user",
            content=[some_text_content_1],
        )

        instance.delete(message_id=created.id, thread_id="A")
        result = instance.retreive(message_id=created.id, thread_id="A")

        assert result is None

    def test_deleted_message_return_id(self, instance):
        created = instance.create(
            thread_id="A",
            role="user",
            content=[some_text_content_1],
        )

        result = instance.delete(message_id=created.id, thread_id="A")

        assert result.id is created.id
        assert result.deleted is True


class TestListInMemoryMessageRepository:
    @pytest.fixture
    def instance(self):
        return InMemoryMessageRepository(data={})

    def test_messages_of_a_thread_are_retreivable_by_page(
        self, instance: InMemoryMessageRepository
    ):
        message_a = instance.create(
            thread_id="A",
            role="user",
            content=[some_text_content_1],
        )
        message_b = instance.create(
            thread_id="A",
            role="assistant",
            content=[some_text_content_2],
        )

        result = instance.listByPage(thread_id="A").data

        assert len(result) == 2
        assert result[0].id == message_a.id
        assert result[1].id == message_b.id

    def test_messages_of_a_thread_are_retreivable(
        self, instance: InMemoryMessageRepository
    ):
        message_a = instance.create(
            thread_id="A",
            role="user",
            content=[some_text_content_1],
        )
        message_b = instance.create(
            thread_id="A",
            role="assistant",
            content=[some_text_content_2],
        )

        result = instance.list(thread_id="A")

        assert len(result) == 2
        assert result[0].id == message_a.id
        assert result[1].id == message_b.id

    def test_messages_of_a_another_thread_are_not_listed(
        self, instance: InMemoryMessageRepository
    ):
        message_a = instance.create(
            thread_id="A",
            role="user",
            content=[some_text_content_1],
        )
        message_b = instance.create(
            thread_id="B",
            role="assistant",
            content=[some_text_content_2],
        )

        result_thread_a = instance.list(thread_id="A")
        result_thread_b = instance.list(thread_id="B")

        assert len(result_thread_a) == 1
        assert result_thread_a[0].id == message_a.id
        assert len(result_thread_b) == 1
        assert result_thread_b[0].id == message_b.id
