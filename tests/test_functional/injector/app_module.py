from injector import Binder, Module, singleton

from langchain_openai_api_bridge.assistant import (
    ThreadRepository,
    MessageRepository,
    RunRepository,
    InMemoryThreadRepository,
    InMemoryMessageRepository,
    InMemoryRunRepository,
)
from langchain_openai_api_bridge.core import AgentFactory
from tests.test_functional.injector.with_injector_my_agent_factory import (
    WithInjectorMyAgentFactory,
)


class MyAppModule(Module):
    def configure(self, binder: Binder):
        binder.bind(ThreadRepository, to=InMemoryThreadRepository, scope=singleton)
        binder.bind(MessageRepository, to=InMemoryMessageRepository, scope=singleton)
        binder.bind(RunRepository, to=InMemoryRunRepository, scope=singleton)
        binder.bind(AgentFactory, to=WithInjectorMyAgentFactory)
