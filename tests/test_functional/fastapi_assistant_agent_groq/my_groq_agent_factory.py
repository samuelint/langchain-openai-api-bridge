from langchain_groq import ChatGroq
from langchain_openai_api_bridge.core.agent_factory import AgentFactory
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

from langchain_openai_api_bridge.core.create_agent_dto import CreateAgentDto


class MyGroqAgentFactory(AgentFactory):

    def create_agent(self, llm: BaseChatModel, dto: CreateAgentDto) -> Runnable:
        return create_react_agent(
            llm,
            [],
            messages_modifier="""You are a helpful assistant.""",
        )

    def create_llm(self, dto: CreateAgentDto) -> Runnable:
        chat_model = ChatGroq(
            model=dto.model,
            streaming=False,  # Must be set to false. Groq does not support tool calling with stream at the moment (23/07/2024)
        )

        return chat_model
