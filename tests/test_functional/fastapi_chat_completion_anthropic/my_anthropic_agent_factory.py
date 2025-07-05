from langchain_openai_api_bridge.core.base_agent_factory import BaseAgentFactory
from langchain_core.runnables import Runnable
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic

from langchain_openai_api_bridge.core.create_agent_dto import CreateAgentDto


class MyAnthropicAgentFactory(BaseAgentFactory):

    def create_agent(self, dto: CreateAgentDto) -> Runnable:
        llm = ChatAnthropic(
            model=dto.model,
            streaming=True,
        )

        return create_react_agent(
            llm,
            [],
            prompt="""You are a helpful assistant.""",
        )
