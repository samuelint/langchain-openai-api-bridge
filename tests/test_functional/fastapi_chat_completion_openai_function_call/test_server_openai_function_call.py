import json
import pytest
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam
from openai.lib.streaming.chat import ChatCompletionStreamState
from fastapi.testclient import TestClient
from server_openai_function_call import app


test_api = TestClient(app)


@pytest.fixture
def openai_client():
    return OpenAI(
        base_url="http://testserver/my-custom-path/openai/v1",
        http_client=test_api,
    )


@pytest.fixture
def tools():
    return [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country e.g. Bogotá, Colombia"
                    }
                },
                "required": [
                    "location"
                ],
                "additionalProperties": False
            },
            "strict": True
        }
    }]


def test_chat_completion_function_call_weather(openai_client: OpenAI, tools: list[ChatCompletionToolParam]):
    chat_completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": 'What is the weather like in London today?',
            }
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )

    assert chat_completion.choices[0].finish_reason == "tool_calls"
    assert chat_completion.choices[0].message.tool_calls[0].function.name == "get_weather"

    args = json.loads(chat_completion.choices[0].message.tool_calls[0].function.arguments)
    assert "london" in args["location"].lower()


def test_chat_completion_function_call_weather_stream(openai_client: OpenAI, tools: list[ChatCompletionToolParam]):
    chunks = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": 'What is the weather like in London today?',
            }
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        stream=True,
    )

    state = ChatCompletionStreamState()
    for chunk in chunks:
        state.handle_chunk(chunk)

    chat_completion = state.get_final_completion()

    assert chat_completion.choices[0].finish_reason == "tool_calls"
    assert chat_completion.choices[0].message.function_call.name == "get_weather"
    assert chat_completion.choices[0].message.role == "assistant"

    args = json.loads(chat_completion.choices[0].message.function_call.arguments)
    assert "london" in args["location"].lower()


def test_chat_completion_function_call_not_called(openai_client: OpenAI, tools: list[ChatCompletionToolParam]):
    chat_completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": 'Hello!',
            }
        ],
        tools=tools,
        tool_choice="none",
    )

    assert chat_completion.choices[0].finish_reason == "stop"
