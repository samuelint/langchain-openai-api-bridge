from abc import ABC, abstractmethod
from langchain_core.runnables import Runnable
from langchain_openai_api_bridge.core.create_agent_dto import CreateAgentDto


class BaseAgentFactory(ABC):

    @abstractmethod
    def create_agent(self, dto: CreateAgentDto) -> Runnable:
        pass
