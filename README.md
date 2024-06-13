# Langchain Openai API Bridge

🚀 Expose [Langchain](https://github.com/langchain-ai/langchain) Agent ([Langgraph](https://github.com/langchain-ai/langgraph)) result as an OpenAI-compatible API 🚀

A `FastAPI` + `Langchain` / `langgraph` extension to expose agent result as an OpenAI-compatible API.

Use any OpenAI-compatible UI or UI framework with your custom `Langchain Agent`.

Support:

- ✅ [Chat Completions API](https://platform.openai.com/docs/api-reference/chat)
  - ✅ Invoke
  - ✅ Stream
- ✅ [Assistant API](https://platform.openai.com/docs/api-reference/assistants) - Feature in progress
  - ✅ Run Stream
  - ✅ Threads
  - ✅ Messages
  - ✅ Run
  - ✅ Tools step stream
  - 🚧 Human In The Loop

If you find this project useful, please give it a star ⭐!

## Table of Content

- [Quick Install](#quick-install)
- [Usage](#usage)
  - [OpenAI Assistant API Compatible](#openai-assistant-api-compatible)
  - [OpenAI Chat Completion API Compatible](#openai-chat-completion-api-compatible)
- [More Examples](#more-examples)
- [Contributing](#contributing)
  - [Installation](#installation)
  - [Commands](#commands)
- [Limitations](#limitations)

## Quick Install

##### pip

```bash
pip install langchain-openai-api-bridge
```

##### poetry

```bash
poetry add langchain-openai-api-bridge
```

## Usage

### OpenAI Assistant API Compatible

```python
# Assistant Bridge as OpenAI Compatible API
assistant_app = AssistantApp(
    thread_repository_type=InMemoryThreadRepository,
    message_repository_type=InMemoryMessageRepository,
    run_repository=InMemoryRunRepository,
    agent_factory=MyAgentFactory,
)

api = FastAPI(
    title="Langchain Agent OpenAI API Bridge",
    version="1.0",
    description="OpenAI API exposing langchain agent",
)

include_assistant(app=api, assistant_app=assistant_app, prefix="/assistant")

if __name__ == "__main__":
    uvicorn.run(api, host="localhost")
```

```python
# Agent Creation
@tool
def magic_number_tool(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2


class MyAgentFactory(AgentFactory):

    def create_agent(self, llm: BaseChatModel) -> CompiledGraph:
        return create_react_agent(
            llm,
            [magic_number_tool],
            messages_modifier="""You are a helpful assistant.""",
        )

    def create_llm(self, dto: CreateLLMDto) -> CompiledGraph:
        return ChatOpenAI(
            model=dto.model,
            api_key=dto.api_key,
            streaming=True,
            temperature=dto.temperature,
        )
```

Full example:

- [Server](tests/test_functional/fastapi_assistant_agent_openai_advanced/assistant_server_openai.py)
- [Agent Factory](tests/test_functional/fastapi_assistant_agent_openai_advanced/my_agent_factory.py)
- [Client](tests/test_functional/fastapi_assistant_agent_openai_advanced/test_assistant_server_openai.py)

### OpenAI Chat Completion API Compatible

```python
# Server
api = FastAPI(
    title="Langchain Agent OpenAI API Bridge",
    version="1.0",
    description="OpenAI API exposing langchain agent",
)

@tool
def magic_number_tool(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2


def assistant_openai_v1_chat(request: OpenAIChatCompletionRequest, api_key: str):
    llm = ChatOpenAI(
        model=request.model,
        api_key=api_key,
        streaming=True,
    )
    agent = create_react_agent(
        llm,
        [magic_number_tool],
        messages_modifier="""You are a helpful assistant.""",
    )

    return V1ChatCompletionRoutesArg(model_name=request.model, agent=agent)


add_v1_chat_completions_agent_routes(
    api,
    path="/my-custom-path",
    handler=assistant_openai_v1_chat,
    system_fingerprint=system_fingerprint,
)

```

```python
# Client
openai_client = OpenAI(
    base_url="http://my-server/my-custom-path/openai/v1",
)

chat_completion = openai_client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": 'Say "This is a test"',
        }
    ],
)
print(chat_completion.choices[0].message.content)
#> "This is a test"
```

Full example:

- [Server](tests/test_functional/fastapi_chat_completion_openai/server_openai.py)
- [Client](tests/test_functional/fastapi_chat_completion_openai/test_server_openai.py)

```typescript
// Vercel AI sdk - example
// ************************
// app/api/my-chat/route.ts
import { NextRequest } from "next/server";
import { z } from "zod";
import { type CoreMessage, streamText } from "ai";
import { createOpenAI } from "@ai-sdk/openai";

export const ChatMessageSchema = z.object({
  id: z.string(),
  role: z.string(),
  createdAt: z.date().optional(),
  content: z.string(),
});

const BodySchema = z.object({
  messages: z.array(ChatMessageSchema),
});

export type AssistantStreamBody = z.infer<typeof BodySchema>;

const langchain = createOpenAI({
  //baseURL: "https://my-project/my-custom-path/openai/v1",
  baseURL: "http://localhost:8000/my-custom-path/openai/v1",
});

export async function POST(request: NextRequest) {
  const { messages }: { messages: CoreMessage[] } = await request.json();

  const result = await streamText({
    model: langchain("gpt-4o"),
    messages,
  });

  return result.toAIStreamResponse();
}
```

## More Examples

Every examples can be found in [`tests/test_functional`](tests/test_functional) directory.

- **OpenAI LLM -> Langgraph Agent -> OpenAI Completion** - [Server](tests/test_functional/fastapi_chat_completion_openai/server_openai.py), [Client](tests/test_functional/fastapi_chat_completion_openai/test_server_openai.py)
- **Anthropic LLM -> Langgraph Agent -> OpenAI Completion** - [Server](tests/test_functional/fastapi_chat_completion_anthropic/server_anthropic.py), [Client](tests/test_functional/fastapi_chat_completion_anthropic/test_server_anthropic.py)
- **Advanced** - OpenAI LLM -> Langgraph Agent -> OpenAI Completion - [Server](tests/test_functional/fastapi_chat_completion_agent_simple/server_openai_advanced.py), [Client](tests/test_functional/fastapi_chat_completion_agent_simple/test_server_openai_advanced.py)

##### ⚠️ Setup to run examples

Define `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` on your system.
Examples will take token from environment variable or `.env` at root of the project.

## Contributing

If you want to contribute to this project, you can follow this guideline:

1. Fork this project
2. Create a new branch
3. Implement your feature or bug fix
4. Send a pull request

### Installation

```sh
poetry install
poetry env use ./.venv/bin/python
```

### Commands

| Command   | Command             |
| --------- | ------------------- |
| Run Tests | `poetry run pytest` |

## Limitations

- **Chat Completions Tools**

  - Functions cannot be passed through open ai API. Every functions need to be defined as a tool in langchain. [Usage Example](tests/test_functional/fastapi_chat_completion_openai/server_openai.py)

- **LLM Usage Info**
  - **Returned usage info is innacurate**. This is due to a Langchain/Langgraph limitation where usage info isn't available when calling a Langgraph Agent.
