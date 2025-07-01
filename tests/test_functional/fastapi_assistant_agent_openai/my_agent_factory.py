from langchain_openai_api_bridge.core.base_agent_factory import BaseAgentFactory
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool, StructuredTool
from langchain_openai_api_bridge.core.create_agent_dto import CreateAgentDto


@tool
def magic_number_tool(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2


class JokerToolFactory:
    def create_as_tool(self, dto: CreateAgentDto) -> BaseTool:
        runnable = self._create_runnable(dto)

        def call_joker_tool(query: str) -> str:
            result = runnable.invoke(query)

            return result.content

        return StructuredTool.from_function(
            func=call_joker_tool,
            name="Joker",
            description="Tell jokes.",
        )

    def _create_runnable(self, dto: CreateAgentDto):
        joker_llm = ChatOpenAI(
            model=dto.model,
            api_key=dto.api_key,
            streaming=True,
            temperature=dto.temperature,
        )
        prompt = PromptTemplate.from_template(
            "You tell jokes. No matter the question. You have to tell a joke."
            "{messages}"
        )

        return prompt | joker_llm


class MyAgentFactory(BaseAgentFactory):

    def create_agent(self, dto: CreateAgentDto) -> Runnable:
        main_llm = ChatOpenAI(
            model=dto.model,
            api_key=dto.api_key,
            streaming=True,
            temperature=dto.temperature,
        )

        joker_tool = JokerToolFactory().create_as_tool(dto)

        return create_react_agent(
            main_llm,
            [magic_number_tool, joker_tool],
            prompt="""You are a helpful assistant.""",
        )
