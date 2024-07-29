from langchain_openai_api_bridge.assistant.create_thread_runs_api_dto import (
    ThreadRunsDto,
)
from langchain_openai_api_bridge.core.base_agent_factory import BaseAgentFactory
from langchain_openai_api_bridge.core.create_agent_dto import CreateAgentDto
from langchain_core.runnables import Runnable


class InternalAgentFactory:
    def __init__(self, agent_factory: BaseAgentFactory) -> None:
        self.agent_factory = agent_factory

    def create_agent(self, thread_run_dto: ThreadRunsDto, api_key: str) -> Runnable:
        create_agent_dto = CreateAgentDto(
            model=thread_run_dto.model,
            thread_id=thread_run_dto.thread_id,
            api_key=api_key,
            temperature=thread_run_dto.temperature,
            assistant_id=thread_run_dto.assistant_id,
        )
        return self.agent_factory.create_agent(dto=create_agent_dto)
