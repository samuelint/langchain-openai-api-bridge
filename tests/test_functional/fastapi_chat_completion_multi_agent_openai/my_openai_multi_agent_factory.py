import functools
import operator
from datetime import datetime
from typing import TypedDict, Annotated, Sequence
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from langchain_openai_api_bridge.core.base_agent_factory import BaseAgentFactory
from langchain_openai_api_bridge.core.create_agent_dto import CreateAgentDto


# Define a new tool that returns the current datetime
datetime_tool = Tool(
    name="Datetime",
    func=lambda x: datetime.now().isoformat(),
    description="Returns the current datetime",
)

mock_search_tool = Tool(
    name="Search",
    func=lambda x: "light",
    description="Search the web about something",
)


def create_agent(llm: ChatOpenAI, system_prompt: str, tools: list):
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


members = ["Researcher", "CurrentTime"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
# Our team supervisor is an LLM node. It just picks the next agent to process and decides when the work is completed
options = ["FINISH"] + members

# Using openai function calling can make output parsing easier for us
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}

# Create the prompt using ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))


# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str


def create_graph(llm):
    # Construction of the chain for the supervisor agent
    supervisor_chain = (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )

    # Add the research agent using the create_agent helper function
    research_agent = create_agent(llm, "You are a web researcher.", [mock_search_tool])
    research_node = functools.partial(
        agent_node, agent=research_agent, name="Researcher"
    )

    # Add the time agent using the create_agent helper function
    current_time_agent = create_agent(
        llm, "You can tell the current time at", [datetime_tool]
    )
    current_time_node = functools.partial(
        agent_node, agent=current_time_agent, name="CurrentTime"
    )

    workflow = StateGraph(AgentState)

    # Add a "chatbot" node. Nodes represent units of work. They are typically regular python functions.
    workflow.add_node("Researcher", research_node)
    workflow.add_node("CurrentTime", current_time_node)
    workflow.add_node("supervisor", supervisor_chain)

    # We want our workers to ALWAYS "report back" to the supervisor when done
    for member in members:
        workflow.add_edge(member, "supervisor")

    # Conditional edges usually contain "if" statements to route
    # to different nodes depending on the current graph state.
    # These functions receive the current graph state and return a string
    # or list of strings indicating which node(s) to call next.
    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

    # Add an entry point. This tells our graph where to start its work each time we run it.
    workflow.add_edge(START, "supervisor")

    # To be able to run our graph, call "compile()" on the graph builder.
    # This creates a "CompiledGraph" we can use invoke on our state.
    graph = workflow.compile(debug=True).with_config(
        RunnableConfig(
            recursion_limit=10,
        )
    )

    return graph


class MyOpenAIMultiAgentFactory(BaseAgentFactory):

    def create_agent(self, dto: CreateAgentDto) -> Runnable:
        llm = ChatOpenAI(
            model=dto.model,
            api_key=dto.api_key,
            streaming=True,
            temperature=dto.temperature,
        )
        return create_graph(llm)
