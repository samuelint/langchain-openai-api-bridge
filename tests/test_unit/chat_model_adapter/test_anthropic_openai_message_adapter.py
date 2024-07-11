from langchain_openai_api_bridge.chat_model_adapter.anthropic_openai_message_adapter import (
    AnthropicOpenAIChatMessageAdapter,
)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


class TestToOpenAIFormatMessages:
    def test_string_content_is_unchanged(self):
        result = AnthropicOpenAIChatMessageAdapter.to_openai_format_messages(
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

    def test_content_in_array_format_is_preserved(self):
        result = AnthropicOpenAIChatMessageAdapter.to_openai_format_messages(
            [
                [
                    SystemMessage(content="Your are an assistant"),
                    HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": "What is in the image?",
                            },
                        ]
                    ),
                ],
            ]
        )

        assert result[0][1].type == "human"
        assert result[0][1].content[0]["type"] == "text"
        assert result[0][1].content[0]["text"] == "What is in the image?"

    def test_base64_image_url_is_converted_to_anthropic_format(self):
        result = AnthropicOpenAIChatMessageAdapter.to_openai_format_messages(
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
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,iVBORw0KGgo="
                                },
                            },
                        ]
                    ),
                ],
            ]
        )

        assert result[0][1].content[1]["type"] == "image"
        assert result[0][1].content[1]["source"]["type"] == "base64"
        assert result[0][1].content[1]["source"]["media_type"] == "image/png"
        assert result[0][1].content[1]["source"]["data"] == "iVBORw0KGgo="
