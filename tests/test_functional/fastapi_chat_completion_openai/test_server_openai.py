import pytest
from openai import OpenAI
from fastapi.testclient import TestClient
from server_openai import api


test_api = TestClient(api)


@pytest.fixture
def openai_client():
    return OpenAI(
        base_url="http://testserver/my-custom-path/openai/v1",
        http_client=test_api,
    )


def test_chat_completion_invoke(openai_client):
    chat_completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
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
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": 'Say "This is a test"'}],
        stream=True,
    )
    every_content = []
    for chunk in chunks:
        if chunk.choices and isinstance(chunk.choices[0].delta.content, str):
            every_content.append(chunk.choices[0].delta.content)

    stream_output = "".join(every_content)

    assert "This is a test" in stream_output


def test_tool(openai_client):

    chat_completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": 'Say "Magic number of 2"',
            }
        ],
    )
    assert "4" in chat_completion.choices[0].message.content
