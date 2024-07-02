from typing import Callable, Optional, Union
from langchain_openai_api_bridge.core.agent_factory import AgentFactory
from langchain_openai_api_bridge.core.utils import TinyDIContainer


class LangchainOpenaiApiBridge:
    def __init__(
        self,
        agent_factory_provider: Union[Callable[[], AgentFactory], AgentFactory],
        tiny_di_container: Optional[TinyDIContainer] = None,
    ) -> None:
        self.tiny_di_container = tiny_di_container or TinyDIContainer()

        self.tiny_di_container.register(AgentFactory, to=agent_factory_provider)
