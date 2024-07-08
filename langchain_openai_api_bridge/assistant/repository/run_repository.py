from openai.types.beta.threads import Run
from abc import ABC, abstractmethod
from typing import List, Literal, Optional
from openai.types.beta.threads.run import RequiredAction, RunStatus, AssistantTool
from openai.pagination import SyncCursorPage


class RunRepository(ABC):

    def __init__(
        self,
    ) -> None:
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def update(self, run: Run) -> Run:
        pass

    @abstractmethod
    def retreive(self, run_id: str) -> Run:
        pass

    @abstractmethod
    def list(self, thread_id: str) -> Run:
        pass

    @abstractmethod
    def listByPage(
        self,
        thread_id: str,
        after: str = None,
        before: str = None,
        limit: int = None,
        order: Literal["asc", "desc"] = None,
    ) -> SyncCursorPage[Run]:
        pass

    @abstractmethod
    def delete(self, run: Optional[Run], run_id: Optional[str]) -> None:
        pass

    @abstractmethod
    def delete_with_thread_id(self, thread_id: str) -> None:
        pass
