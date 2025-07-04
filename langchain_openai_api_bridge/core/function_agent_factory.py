import inspect
from typing import Callable, Union, Awaitable
from langchain_core.runnables import Runnable
from langchain_openai_api_bridge.core.base_agent_factory import BaseAgentFactory
from langchain_openai_api_bridge.core.create_agent_dto import CreateAgentDto


class FunctionAgentFactory(BaseAgentFactory):

    def __init__(
        self,
        fn: Union[
            Callable[[CreateAgentDto], Runnable],
            Callable[[CreateAgentDto], Awaitable[Runnable]]
        ],
    ) -> None:
        self.fn = fn
        self.is_async = inspect.iscoroutinefunction(fn)

    async def acreate_agent(self, dto: CreateAgentDto) -> Runnable:
        if self.is_async:
            return await self.fn(dto)
        else:
            return self.fn(dto)
