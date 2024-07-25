from langchain_openai_api_bridge.core.agent_factory import AgentFactory
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic

from langchain_openai_api_bridge.core.create_agent_dto import CreateAgentDto


class MyAnthropicAgentFactory(AgentFactory):

    def create_agent(self, llm: BaseChatModel, dto: CreateAgentDto) -> Runnable:
        return create_react_agent(
            llm,
            [],
            messages_modifier="""You are a helpful assistant.""",
        )

    def create_llm(self, dto: CreateAgentDto) -> Runnable:
        return ChatAnthropic(
            model=dto.model,
            streaming=True,
        )
