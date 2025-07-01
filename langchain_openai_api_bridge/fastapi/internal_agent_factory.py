from langchain_openai_api_bridge.assistant.create_thread_runs_api_dto import (
    ThreadRunsDto,
)
from langchain_openai_api_bridge.core.base_agent_factory import BaseAgentFactory, PotentiallyRunnable, wrap_agent
from langchain_openai_api_bridge.core.create_agent_dto import CreateAgentDto
from langchain_core.runnables import Runnable

from typing import (
    AsyncContextManager,
)


class InternalAgentFactory:
    def __init__(self, agent_factory: BaseAgentFactory) -> None:
        self.agent_factory = agent_factory

    def create_agent(
        self, thread_run_dto: ThreadRunsDto, api_key: str
    ) -> PotentiallyRunnable:
        create_agent_dto = CreateAgentDto(
            model=thread_run_dto.model,
            thread_id=thread_run_dto.thread_id,
            api_key=api_key,
            temperature=thread_run_dto.temperature,
            assistant_id=thread_run_dto.assistant_id,
        )
        return self.agent_factory.create_agent(dto=create_agent_dto)

    def create_agent_with_async_context(
        self, thread_run_dto: ThreadRunsDto, api_key: str
    ) -> AsyncContextManager[Runnable]:
        return wrap_agent(self.create_agent(thread_run_dto, api_key))
