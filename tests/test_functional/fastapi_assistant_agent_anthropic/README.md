# Expose as an OpenAI API anthropic assistant (Assistant API)

The default `from langchain_anthropic import ChatAnthropic` is not compatible with multimodal prompts as the image format differs between OpenAI and Anthropic.

To use multimodal prompts, use the `OpenAICompatibleChatModel` which transforms OpenAI format to Anthropic format. This enables you to use one or the other seamlessly.
Look at `my_anthropic_agent_factory.py` for usage example.

```python
chat_model = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    max_tokens=1024,
    streaming=True,
)

return OpenAICompatibleChatModel(chat_model=chat_model)

```

#### Multimodal Formats differences

##### Anthropic

https://docs.anthropic.com/en/docs/build-with-claude/vision#about-the-prompt-examples

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
