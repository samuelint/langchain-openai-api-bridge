from abc import ABC, abstractmethod
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langchain_openai_api_bridge.core.create_agent_dto import CreateAgentDto


class AgentFactory(ABC):

    @abstractmethod
    def create_agent(self, llm: BaseChatModel, dto: CreateAgentDto) -> Runnable:
        pass

    @abstractmethod
    def create_llm(self, dto: CreateAgentDto) -> BaseChatModel:
        pass
