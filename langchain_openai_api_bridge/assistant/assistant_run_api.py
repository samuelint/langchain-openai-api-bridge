from openai import Stream
from openai.types.beta.threads import Run
from openai.types.beta import AssistantStreamEvent
from abc import ABC, abstractmethod


class AssistantRunAPI(ABC):

    def __init__(
        self,
    ) -> None:
        pass

    @abstractmethod
    def run(self, thread_id: str) -> Run:
        pass

    @abstractmethod
    def stream(self, thread_id: str) -> Stream[AssistantStreamEvent]:
        pass
