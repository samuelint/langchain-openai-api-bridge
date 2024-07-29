import multiprocessing
import os

from langchain_openai_api_bridge.chat_model_adapter.llamacpp import (
    LLamacppOpenAICompatibleChatModel,
)
from langchain_openai_api_bridge.core.base_agent_factory import BaseAgentFactory
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from langchain_openai_api_bridge.core.create_agent_dto import CreateAgentDto
from llama_cpp import Llama


@tool
def magic_number_tool(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2


class MyLlamacppAgentFactory(BaseAgentFactory):

    def create_agent(self, dto: CreateAgentDto) -> Runnable:
        llm = self.create_llm(dto=dto)

        return create_react_agent(
            llm,
            [magic_number_tool],
            # messages_modifier="""You are a helpful assistant.""",
        )

    def create_llm(self, dto: CreateAgentDto) -> BaseChatModel:
        model_path = os.path.join(
            os.path.expanduser("~/.cache/lm-studio/models"),
            "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        )

        llama = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            offload_kqv=True,  # Equivalent of f16_kv=True
            n_threads=multiprocessing.cpu_count() - 1,
            chat_format="chatml-function-calling",
        )

        return LLamacppOpenAICompatibleChatModel(
            llama=llama,
            temperature=dto.temperature or 0,
            streaming=True,
        )
