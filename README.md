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

- [Server Code](tests/test_functional/fastapi_chat_completion_openai/server.py)
- [Client Code ](tests/test_functional/fastapi_chat_completion_openai/test_server.py)

##### 2. FastAPI endpoint expose Langchain (Langgraph) Agent as an OpenAI chat completion api using anthropic model

- [Server Code](tests/test_functional/fastapi_chat_completion_anthropic/server.py)
- [Client Code ](tests/test_functional/fastapi_chat_completion_anthropic/test_server.py)

## Contribute

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
