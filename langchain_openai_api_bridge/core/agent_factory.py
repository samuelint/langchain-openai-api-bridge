from abc import ABC, abstractmethod
from langgraph.graph.graph import CompiledGraph
from langchain_core.language_models import BaseChatModel
from langchain_openai_api_bridge.core.create_agent_dto import CreateAgentDto


class AgentFactory(ABC):

    @abstractmethod
    def create_agent(self, llm: BaseChatModel, dto: CreateAgentDto) -> CompiledGraph:
        pass

    @abstractmethod
    def create_llm(self, dto: CreateAgentDto) -> CompiledGraph:
        pass
