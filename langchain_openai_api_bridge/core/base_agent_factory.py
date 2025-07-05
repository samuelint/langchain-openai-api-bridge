from abc import ABC, abstractmethod
from langchain_core.runnables import Runnable
from langchain_openai_api_bridge.core.create_agent_dto import CreateAgentDto
from typing import (
    Awaitable,
    Generator,
    ContextManager,
    AsyncContextManager,
    AsyncGenerator,
    Union,
)
import inspect
from contextlib import asynccontextmanager

PotentiallyRunnable = Union[
    Runnable,
    Awaitable[Runnable],
    Generator[Runnable, None, None],
    ContextManager[Runnable],
    AsyncGenerator[Runnable],
    AsyncContextManager[Runnable],
]


class BaseAgentFactory(ABC):

    @abstractmethod
    def create_agent(self, dto: CreateAgentDto) -> PotentiallyRunnable:
        pass

    def create_agent_with_async_context(
        self, dto: CreateAgentDto
    ) -> AsyncContextManager[Runnable]:
        return wrap_agent(self.create_agent(dto))


@asynccontextmanager
async def _agen_wrapper(agen: AsyncGenerator[Runnable]):
    try:
        yield await agen.__anext__()
    finally:
        async for _ in agen:
            pass


@asynccontextmanager
async def _sync_wrapper(cm):
    with cm as value:
        yield value


@asynccontextmanager
async def _gen_wrapper(gen: Generator[Runnable, None, None]):
    try:
        yield next(gen)
    finally:
        for _ in gen:
            pass


@asynccontextmanager
async def _coro_wrapper(coro: Awaitable[Runnable]):
    yield await coro


@asynccontextmanager
async def _value_wrapper(value: Runnable):
    yield value


def wrap_agent(agent: PotentiallyRunnable):
    if hasattr(agent, "__aenter__") and hasattr(agent, "__aexit__"):
        return agent

    if inspect.isasyncgen(agent):
        return _agen_wrapper(agent)

    if hasattr(agent, "__enter__") and hasattr(agent, "__exit__"):
        return _sync_wrapper(agent)

    if inspect.isgenerator(agent):
        return _gen_wrapper(agent)

    if inspect.iscoroutine(agent):
        return _coro_wrapper(agent)

    return _value_wrapper(agent)
