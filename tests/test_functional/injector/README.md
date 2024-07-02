# With Injector Example

Usage example using a third party injector library.
`poetry add injector` https://github.com/python-injector/injector for this example

```python
# with_injector_assistant_server_openai.py (main)
from fastapi import FastAPI
from dotenv import load_dotenv, find_dotenv
import uvicorn
from injector import Injector

from langchain_openai_api_bridge.assistant import (
    ThreadRepository,
    MessageRepository,
    RunRepository,
)
from langchain_openai_api_bridge.core.agent_factory import AgentFactory
from langchain_openai_api_bridge.fastapi import (
    LangchainOpenaiApiBridgeFastAPI,
)
from tests.test_functional.injector.app_module import MyAppModule


_ = load_dotenv(find_dotenv())


app = FastAPI(
    title="Langchain Agent OpenAI API Bridge",
    version="1.0",
    description="OpenAI API exposing langchain agent using injector",
)

injector = Injector([MyAppModule()])

bridge = LangchainOpenaiApiBridgeFastAPI(
    app=app, agent_factory_provider=lambda: injector.get(AgentFactory)
)
bridge.bind_openai_assistant_api(
    thread_repository_provider=lambda: injector.get(ThreadRepository),
    message_repository_provider=lambda: injector.get(MessageRepository),
    run_repository_provider=lambda: injector.get(RunRepository),
    prefix="/my-assistant",
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost")

```

```python
# app_module.py
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

```
