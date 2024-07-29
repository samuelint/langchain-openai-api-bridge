from typing import Callable, Optional, Union
from langchain_openai_api_bridge.core.base_agent_factory import BaseAgentFactory
from langchain_openai_api_bridge.core.utils import TinyDIContainer


class LangchainOpenaiApiBridge:
    def __init__(
        self,
        agent_factory_provider: Union[Callable[[], BaseAgentFactory], BaseAgentFactory],
        tiny_di_container: Optional[TinyDIContainer] = None,
    ) -> None:
        self.tiny_di_container = tiny_di_container or TinyDIContainer()

        self.tiny_di_container.register(BaseAgentFactory, to=agent_factory_provider)
