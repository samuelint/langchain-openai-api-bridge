from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langchain_openai_api_bridge.chat_model_adapter.llamacpp.llamacpp_openai_compatible_chat_model_adapter import (
    LlamacppOpenAICompatibleChatModelAdapter,
)

instance = LlamacppOpenAICompatibleChatModelAdapter()


class TestToOpenAIFormatMessages:
    def test_string_content_is_unchanged(self):
        result = instance.to_openai_format_messages(
            [
                [
                    SystemMessage(content="Your are an assistant"),
                    HumanMessage(content="Hi, I am human"),
                    AIMessage(content="Hi, human. I am an AI"),
                ],
            ]
        )

        assert result[0][0].type == "system"
        assert result[0][0].content == "Your are an assistant"
        assert result[0][1].type == "human"
        assert result[0][1].content == "Hi, I am human"
        assert result[0][2].type == "ai"
        assert result[0][2].content == "Hi, human. I am an AI"

    def test_content_in_array_format_is_transformed_to_single_string(self):
        result = instance.to_openai_format_messages(
            [
                [
                    SystemMessage(content="Your are an assistant"),
                    HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": "What is in the image?",
                            },
                            {
                                "type": "text",
                                "text": "And this ?",
                            },
                        ]
                    ),
                ],
            ]
        )

        assert result[0][1].type == "human"
        assert result[0][1].content == "What is in the image?\nAnd this ?"
