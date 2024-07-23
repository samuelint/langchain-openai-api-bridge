from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseChatModel
from langgraph.graph.graph import CompiledGraph
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.messages import AIMessage, BaseMessage


class AgentState(TypedDict):
    """
    Represents the state of our graph.
    """

    messages: list[BaseMessage]


class SimpleAgentBuilder:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    def build(self) -> CompiledGraph:
        workflow = StateGraph(AgentState)
        llm = self.llm

        def call_model(
            state: AgentState,
            config: RunnableConfig,
        ):
            messages = state["messages"]
            response = llm.invoke(messages, config)
            if state["is_last_step"] and response.tool_calls:
                return {
                    "messages": [
                        AIMessage(
                            id=response.id,
                            content="Sorry, need more steps to process this request.",
                        )
                    ]
                }
            # We return a list, because this will get added to the existing list
            return {"messages": [response]}

        async def acall_model(state: AgentState, config: RunnableConfig):
            messages = state["messages"]
            response = await llm.ainvoke(messages, config)
            if state["is_last_step"] and response.tool_calls:
                return {
                    "messages": [
                        AIMessage(
                            id=response.id,
                            content="Sorry, need more steps to process this request.",
                        )
                    ]
                }
            # We return a list, because this will get added to the existing list
            return {"messages": [response]}

        def should_continue(state: AgentState):
            return "end"

        workflow.add_node("agent", RunnableLambda(call_model, acall_model))
        workflow.set_entry_point("agent")

        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "end": END,
            },
        )

        return workflow.compile()
