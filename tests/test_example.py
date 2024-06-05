from langchain_openai_bridge import example


def test_hello():
    assert example.hello() == "Hello, world!"
