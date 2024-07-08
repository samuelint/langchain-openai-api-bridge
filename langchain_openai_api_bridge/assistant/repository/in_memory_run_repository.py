from openai.types.beta.threads import Run
import uuid
from typing import List, Literal, Optional
from langchain_openai_api_bridge.assistant.openai_run_factory import create_run
from langchain_openai_api_bridge.assistant.repository.run_repository import (
    RunRepository,
)
from openai.types.beta.threads.run import RequiredAction, RunStatus, AssistantTool
from openai.pagination import SyncCursorPage


class InMemoryRunRepository(RunRepository):
    def __init__(self, data: Optional[dict[str, Run]] = None) -> None:
        self.runs = data or {}

    def create(
        self,
        assistant_id: str,
        thread_id: str,
        model: str,
        status: RunStatus,
        instructions: str = "",
        required_action: Optional[RequiredAction] = None,
        tools: List[AssistantTool] = [],
        parallel_tool_calls: bool = True,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Run:
        id = str(uuid.uuid4())
        run = create_run(
            id=id,
            assistant_id=assistant_id,
            thread_id=thread_id,
            model=model,
            status=status,
            instructions=instructions,
            required_action=required_action,
            tools=tools,
            parallel_tool_calls=parallel_tool_calls,
            temperature=temperature,
            top_p=top_p,
        )
        self.runs[id] = run

        return self.retreive(run_id=id)

    def retreive(self, run_id: str) -> Run:
        result = self.runs.get(run_id)
        if result is None:
            return None

        return result.copy(deep=True)

    def list(self, thread_id: str) -> Run:
        return [run for run in self.runs.values() if run.thread_id == thread_id]

    def listByPage(
        self,
        thread_id: str,
        after: str = None,
        before: str = None,
        limit: int = None,
        order: Literal["asc", "desc"] = None,
    ) -> SyncCursorPage[Run]:
        runs = self.list(thread_id=thread_id)

        return SyncCursorPage(data=runs)

    def update(self, run: Run) -> Run:
        id = run.id
        self.runs[id] = run
        return self.retreive(run_id=id)

    def delete(self, run: Optional[Run] = None, run_id: Optional[str] = None) -> None:
        if run_id is None:
            if run is None:
                raise ValueError("At least one of run or run_id must be provided")
            id = run.id
        else:
            id = run_id

        del self.runs[id]

    def delete_with_thread_id(self, thread_id: str) -> None:
        runs_to_delete = [
            run for run in self.runs.values() if run.thread_id == thread_id
        ]
        for run in runs_to_delete:
            del self.runs[run.id]
