from typing import Optional, Type
from langchain_openai_api_bridge.assistant.adapter.container import (
    register_assistant_adapter,
)
from langchain_openai_api_bridge.assistant.assistant_message_service import (
    AssistantMessageService,
)
from langchain_openai_api_bridge.assistant.assistant_run_service import (
    AssistantRunService,
)
from langchain_openai_api_bridge.assistant.assistant_thread_service import (
    AssistantThreadService,
)
from langchain_openai_api_bridge.assistant.repository.message_repository import (
    MessageRepository,
)
from langchain_openai_api_bridge.assistant.repository.run_repository import (
    RunRepository,
)
from langchain_openai_api_bridge.assistant.repository.thread_repository import (
    ThreadRepository,
)
from langchain_openai_api_bridge.core.agent_factory import AgentFactory
from langchain_openai_api_bridge.core.utils.di_container import DIContainer


class AssistantApp:
    def __init__(
        self,
        thread_repository_type: Type[ThreadRepository],
        message_repository_type: Type[MessageRepository],
        run_repository: Type[RunRepository],
        agent_factory: Type[AgentFactory],
        system_fingerprint: Optional[str] = "",
    ):
        self.container = DIContainer()
        self.system_fingerprint = system_fingerprint

        register_assistant_adapter(self.container)

        self.container.register(AssistantThreadService)
        self.container.register(AssistantMessageService)
        self.container.register(AssistantRunService)

        self.container.register(
            ThreadRepository, to=thread_repository_type, singleton=True
        )
        self.container.register(
            MessageRepository, to=message_repository_type, singleton=True
        )
        self.container.register(RunRepository, to=run_repository, singleton=True)
        self.container.register(AgentFactory, to=agent_factory)
