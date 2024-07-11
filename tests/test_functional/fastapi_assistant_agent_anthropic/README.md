# Expose as an OpenAI API anthropic assistant (Assistant API)

The default `from langchain_anthropic import ChatAnthropic` is not compatible with multimodal prompts as the image format differs between OpenAI and Anthropic.

To use multimodal prompts, use the `OpenAICompatibleAnthropicChatModel` adapter (from `langchain_openai_api_bridge.chat_model_adapter`) which transforms OpenAI format to Anthropic format. This enables you to use one or the other seamlessly.
Look at `my_anthropic_agent_factory.py` for usage example.

#### Multimodal Formats

##### Anthropic

```python
{
    "role": "user",
    "content": [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": "iVBORw0KGgo=",
            },
        },
        {
            "type": "text",
            "text": "Describe this image."
        }
    ],
}
```

##### OpenAI

```python
{
    "role": "user",
    "content": [
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64,iVBORw0KGgo="
            },
        },
        {
            "type": "text",
            "text": "Describe this image."
        }
    ],
}
```
