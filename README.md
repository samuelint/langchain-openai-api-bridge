# Langchain Openai API Bridge

Use Langchain output as an OpenAI-compatible API.

## Usage

```bash
pip install langchain_openai_bridge
```

```bash
poetry add langchain_openai_bridge
```

### Examples

Most examples can be found in `tests/test_functional`.

##### 1. FastAPI endpoint expose Langchain (Langgraph) Agent as an OpenAI chat completion api

- [Server Code](tests/test_functional/fastapi_chat_completion/server.py)
- [Client Code ](tests/test_functional/fastapi_chat_completion/test_server.py)

## Contribute

### Installation

```sh
poetry install
poetry env use ./.venv/bin/python
```

### Commands

| Command   | Command             |
| --------- | ------------------- |
| Run Tests | `poetry run pytest` |
