from typing import Callable, Union
from fastapi import FastAPI

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
from langchain_openai_api_bridge.core.langchain_openai_api_bridge import (
    LangchainOpenaiApiBridge,
)
from langchain_openai_api_bridge.fastapi.assistant_api_router import (
    create_openai_assistant_router,
)
from langchain_openai_api_bridge.fastapi.chat_completion_router import (
    create_openai_chat_completion_router,
)


class LangchainOpenaiApiBridgeFastAPI(LangchainOpenaiApiBridge):
    def __init__(
        self,
        app: FastAPI,
        agent_factory_provider: Union[Callable[[], AgentFactory], AgentFactory],
    ) -> None:
        super().__init__(agent_factory_provider=agent_factory_provider)
        self.app = app

    def bind_openai_assistant_api(
        self,
        thread_repository_provider: Union[Callable[[], ThreadRepository]],
        message_repository_provider: Union[Callable[[], MessageRepository]],
        run_repository_provider: Union[Callable[[], RunRepository]],
        prefix: str = "",
    ) -> None:

        self.tiny_di_container.register(AssistantThreadService)
        self.tiny_di_container.register(AssistantMessageService)
        self.tiny_di_container.register(AssistantRunService)
        self.tiny_di_container.register(ThreadRepository, to=thread_repository_provider)
        self.tiny_di_container.register(
            MessageRepository, to=message_repository_provider
        )
        self.tiny_di_container.register(RunRepository, to=run_repository_provider)
        register_assistant_adapter(self.tiny_di_container)

        assistant_router = create_openai_assistant_router(
            tiny_di_container=self.tiny_di_container,
            prefix=prefix,
        )

        self.app.include_router(assistant_router)

    def bind_openai_chat_completion(self, prefix: str = "") -> None:
        chat_completion_router = create_openai_chat_completion_router(
            self.tiny_di_container, prefix=prefix
        )

        self.app.include_router(chat_completion_router)
