from typing import Iterable, List, Type, Union
from openai import BaseModel
from openai.types.beta.threads import (
    MessageContentPartParam,
    TextContentBlock,
    ImageFile,
    ImageURL,
    ImageFileContentBlock,
    ImageURLContentBlock,
    Text,
)
from openai.types.beta.threads.message import MessageContent


def to_openai_message_content(
    content: Union[str, MessageContentPartParam] = ""
) -> MessageContent:

    if isinstance(content, str):
        return TextContentBlock(text=Text(value=content, annotations=[]), type="text")
    elif isinstance(content, dict):
        if content["type"] == "text":
            return TextContentBlock(
                type="text",
                text=Text(value=content["text"], annotations=[]),
            )
        if content["type"] == "image_file":
            return ImageFileContentBlock(
                type="image_file",
                image_file=ImageFile(
                    file_id=content["image_file"]["file_id"],
                    detail=content["image_file"]["detail"],
                ),
            )
        if content["type"] == "image_url":
            return ImageURLContentBlock(
                type="image_url",
                image_url=ImageURL(
                    url=content["image_url"]["url"],
                    detail=content["image_url"]["detail"],
                ),
            )
    elif (
        isinstance(content, TextContentBlock)
        or isinstance(content, ImageFileContentBlock)
        or isinstance(content, ImageURLContentBlock)
    ):
        return content


def to_openai_message_content_list(
    content: Union[str, Iterable[MessageContentPartParam]] = ""
) -> List[MessageContent]:
    if isinstance(content, str):
        return [to_openai_message_content(content)]
    elif isinstance(content, Iterable):
        return [to_openai_message_content(item) for item in content]


def deserialize_message_content(data: dict) -> MessageContent:
    type_to_model: dict[str, Type[BaseModel]] = {
        "image_file": ImageFileContentBlock,
        "image_url": ImageURLContentBlock,
        "text": TextContentBlock,
    }

    content_type = data["type"]
    model_cls = type_to_model[content_type]
    return model_cls.parse_obj(data)
