from openai.types.beta.threads import Run
from abc import ABC, abstractmethod
from typing import List, Optional
from openai.types.beta.threads.run import RequiredAction, RunStatus, AssistantTool


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
    def delete(self, run: Optional[Run], run_id: Optional[str]) -> None:
        pass
