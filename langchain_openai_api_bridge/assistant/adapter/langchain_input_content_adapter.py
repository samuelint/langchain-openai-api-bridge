from typing import List
from openai.types.beta.threads.message import MessageContent


def to_langchain_input_content(content: List[MessageContent]) -> list[dict]:
    converted_content = []

    if isinstance(content, str):
        content = [
            {
                "type": "text",
                "text": content,
            }
        ]

    for c in content:
        if c.type == "text":
            converted_content.append(
                {
                    "type": "text",
                    "text": c.text.value,
                }
            )

        if c.type == "image_url":
            converted_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": c.image_url.url},
                }
            )

    return converted_content
