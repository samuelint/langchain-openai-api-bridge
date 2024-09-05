import pytest
from openai import OpenAI
from fastapi.testclient import TestClient
from server_openai import app


test_api = TestClient(app)


@pytest.fixture
def openai_client():
    return OpenAI(
        base_url="http://testserver/my-custom-path/openai/v1",
        http_client=test_api,
    )


def test_chat_completion_invoke(openai_client):
    chat_completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": 'Say "This is a test"',
            }
        ],
    )
    assert "This is a test" in chat_completion.choices[0].message.content


def test_chat_completion_stream(openai_client):
    chunks = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": 'Say "This is a test"'}],
        stream=True,
    )
    every_content = []
    for chunk in chunks:
        if chunk.choices and isinstance(chunk.choices[0].delta.content, str):
            every_content.append(chunk.choices[0].delta.content)

    stream_output = "".join(every_content)

    assert "This is a test" in stream_output

@pytest.fixture
def openai_client_custom_events():
    return OpenAI(
        base_url="http://testserver/my-custom-events-path/openai/v1",
        http_client=test_api,
    )

def test_chat_completion_invoke_custom_events(openai_client_custom_events):
    chat_completion = openai_client_custom_events.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": 'Say "This is a test"',
            }
        ],
    )
    assert "This is a test" in chat_completion.choices[0].message.content


def test_chat_completion_stream_custom_events(openai_client_custom_events):
    chunks = openai_client_custom_events.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": 'Say "This is a test"'}],
        stream=True,
    )
    every_content = []
    for chunk in chunks:
        if chunk.choices and isinstance(chunk.choices[0].delta.content, str):
            every_content.append(chunk.choices[0].delta.content)

    stream_output = "".join(every_content)

    assert "This is a test" in stream_output