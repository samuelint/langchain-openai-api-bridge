from typing import Callable
from langchain_core.runnables import Runnable
from langchain_openai_api_bridge.core.base_agent_factory import BaseAgentFactory
from langchain_openai_api_bridge.core.create_agent_dto import CreateAgentDto


class FunctionAgentFactory(BaseAgentFactory):

    def __init__(
        self,
        fn: Callable[[CreateAgentDto], Runnable],
    ) -> None:
        self.fn = fn

    def create_agent(self, dto: CreateAgentDto) -> Runnable:
        return self.fn(dto)
