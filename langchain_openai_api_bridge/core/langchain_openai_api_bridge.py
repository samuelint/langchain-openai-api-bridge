from typing import Callable
from langchain_openai_api_bridge.core.agent_factory import AgentFactory
from langchain_openai_api_bridge.core.utils import TinyDIContainer


class LangchainOpenaiApiBridge:
    def __init__(
        self,
        agent_factory_provider: Callable[[], AgentFactory],
        tiny_di_container: TinyDIContainer = TinyDIContainer(),
    ) -> None:
        self.tiny_di_container = tiny_di_container

        self.tiny_di_container.register(AgentFactory, to=agent_factory_provider)
