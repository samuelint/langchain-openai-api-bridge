from typing import Optional, Type
from langchain_openai_api_bridge.assistant.adapter.container import (
    register_assistant_adapter,
)
from langchain_openai_api_bridge.assistant.assistant_lib_injector import (
    BaseAssistantLibInjector,
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


class AssistantApp:
    def __init__(
        self,
        injector: BaseAssistantLibInjector,
        thread_repository_type: Optional[Type[ThreadRepository]] = None,
        message_repository_type: Optional[Type[MessageRepository]] = None,
        run_repository: Optional[Type[RunRepository]] = None,
        agent_factory: Optional[Type[AgentFactory]] = None,
        system_fingerprint: Optional[str] = "",
    ):
        self.injector = injector
        self.system_fingerprint = system_fingerprint

        register_assistant_adapter(self.injector)

        self.injector.register(AssistantThreadService)
        self.injector.register(AssistantMessageService)
        self.injector.register(AssistantRunService)

        if thread_repository_type is not None:
            self.injector.register(
                ThreadRepository, to=thread_repository_type, scope="singleton"
            )

        if message_repository_type is not None:
            self.injector.register(
                MessageRepository, to=message_repository_type, scope="singleton"
            )

        if run_repository is not None:
            self.injector.register(RunRepository, to=run_repository, scope="singleton")

        if agent_factory is not None:
            self.injector.register(AgentFactory, to=agent_factory)
