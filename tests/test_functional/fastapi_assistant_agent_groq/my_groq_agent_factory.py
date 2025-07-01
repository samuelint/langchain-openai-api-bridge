from langchain_groq import ChatGroq
from langchain_openai_api_bridge.core.base_agent_factory import BaseAgentFactory
from langchain_core.runnables import Runnable
from langgraph.prebuilt import create_react_agent

from langchain_openai_api_bridge.core.create_agent_dto import CreateAgentDto


class MyGroqAgentFactory(BaseAgentFactory):

    def create_agent(self, dto: CreateAgentDto) -> Runnable:
        llm = ChatGroq(
            model=dto.model,
            streaming=False,  # Must be set to false. Groq does not support tool calling with stream at the moment (23/07/2024)
        )

        return create_react_agent(
            llm,
            [],
            prompt="""You are a helpful assistant.""",
        )
