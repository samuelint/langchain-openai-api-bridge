from openai.types.beta.assistant_stream_event import (
    ThreadRunCreated,
    ThreadRunCompleted,
)
from openai.types.beta.threads import (
    Run,
)

from langchain_openai_api_bridge.assistant.repository.run_repository import (
    RunRepository,
)


class ThreadRunEventHandler:

    def __init__(
        self,
        run_repository: RunRepository,
    ) -> None:
        self.run_repository = run_repository

    def on_thread_run_start(
        self, assistant_id: str, thread_id: str, model: str
    ) -> ThreadRunCreated:
        run = self.run_repository.create(
            assistant_id=assistant_id,
            thread_id=thread_id,
            model=model,
            status="in_progress",
        )

        return ThreadRunCreated(
            event="thread.run.created",
            data=run,
        )

    def on_thread_run_completed(self, run: Run) -> ThreadRunCompleted:
        completed_run = run.copy()
        completed_run.status = "completed"
        completed_run = self.run_repository.update(run=completed_run)

        return ThreadRunCompleted(
            event="thread.run.completed",
            data=completed_run,
        )
