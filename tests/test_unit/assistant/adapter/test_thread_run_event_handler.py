from decoy import Decoy, matchers
import pytest


from langchain_openai_api_bridge.assistant.adapter.thread_run_event_handler import (
    ThreadRunEventHandler,
)
from langchain_openai_api_bridge.assistant.openai_run_factory import create_run

from langchain_openai_api_bridge.assistant.repository.run_repository import (
    RunRepository,
)
from openai.types.beta.threads import (
    Run,
)


@pytest.fixture
def some_run() -> Run:
    return create_run(
        id="run1",
        assistant_id="assistant1",
        thread_id="thread1",
        model="some-model",
        status="in_progress",
    )


@pytest.fixture
def run_repository(decoy: Decoy):
    return decoy.mock(cls=RunRepository)


class TestCreateThreadRun:
    @pytest.fixture
    def instance(self, run_repository: RunRepository):
        instance = ThreadRunEventHandler(run_repository=run_repository)

        return instance

    def test_event_is_created_from_database(
        self,
        decoy: Decoy,
        run_repository: RunRepository,
        instance: ThreadRunEventHandler,
        some_run: Run,
    ):
        decoy.when(
            run_repository.create(
                assistant_id="assistant1",
                thread_id="thread1",
                model="some-model",
                status="in_progress",
                temperature=None,
            )
        ).then_return(some_run)

        result = instance.on_thread_run_start(
            assistant_id="assistant1", thread_id="thread1", model="some-model"
        )

        assert result.data == some_run

    def test_event_type(
        self,
        decoy: Decoy,
        run_repository: RunRepository,
        instance: ThreadRunEventHandler,
        some_run: Run,
    ):
        decoy.when(
            run_repository.create(
                assistant_id="assistant1",
                thread_id="thread1",
                model="some-model",
                status="in_progress",
                temperature=None,
            )
        ).then_return(some_run)

        result = instance.on_thread_run_start(
            assistant_id="assistant1", thread_id="thread1", model="some-model"
        )

        assert result.event == "thread.run.created"


class TestCompleteThreadRun:

    @pytest.fixture
    def instance(self, decoy: Decoy, run_repository: RunRepository):
        instance = ThreadRunEventHandler(run_repository=run_repository)

        decoy.when(run_repository.update(matchers.Anything())).then_do(lambda run: run)

        return instance

    def test_event_type(
        self,
        instance: ThreadRunEventHandler,
        some_run: Run,
    ):
        result = instance.on_thread_run_completed(run=some_run)

        assert result.event == "thread.run.completed"

    def test_event_status_transition_to_completed(
        self,
        instance: ThreadRunEventHandler,
        some_run: Run,
    ):
        some_run.status = "in_progress"

        result = instance.on_thread_run_completed(run=some_run)

        assert result.data.status == "completed"

    def test_run_is_persisted(
        self,
        decoy: Decoy,
        run_repository: RunRepository,
        instance: ThreadRunEventHandler,
        some_run: Run,
    ):
        some_run.status = "in_progress"

        instance.on_thread_run_completed(run=some_run)

        decoy.verify(
            run_repository.update(
                matchers.HasAttributes({"id": "run1", "status": "completed"})
            ),
            times=1,
        )
