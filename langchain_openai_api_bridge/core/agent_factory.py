from abc import ABC, abstractmethod
from langgraph.graph.graph import CompiledGraph
from langchain_core.language_models import BaseChatModel
from langchain_openai_api_bridge.core.create_llm_dto import CreateLLMDto


class AgentFactory(ABC):

    @abstractmethod
    def create_agent(self, llm: BaseChatModel) -> CompiledGraph:
        pass

    @abstractmethod
    def create_llm(self, dto: CreateLLMDto) -> CompiledGraph:
        pass
