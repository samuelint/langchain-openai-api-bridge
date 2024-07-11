# Expose as an OpenAI API Groq assistant (Assistant API)

:warning: Groq does not support streaming with tools. Make sure to set `streaming=False,`

```python
chat_model = ChatGroq(
    model="llama3-8b-8192",
    streaming=False, # <<--- Must be set to False when used with LangGraph / Tools
)
```

:warning: Note that Groq models do not currently support multi-modal capabilities. Do not use payload with image reference

```python
{
    "type": "image_url",
    "image_url": {
        "url": "data:image/jpeg;base64,iVBORw0KGgo="
    },
},
```
