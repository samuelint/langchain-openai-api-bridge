from openai.types.beta.threads import (
    TextContentBlockParam,
    TextContentBlock,
    ImageURLContentBlockParam,
    ImageURLContentBlock,
    ImageURLParam,
    ImageFileContentBlockParam,
    ImageFileContentBlock,
    ImageFileParam,
)
from langchain_openai_api_bridge.assistant.adapter.openai_message_content_adapter import (
    deserialize_message_content,
    to_openai_message_content,
    to_openai_message_content_list,
)


class TestToOpenaiMessageContent:
    def test_string_content_is_converted_to_TextContentBlock(self):
        result = to_openai_message_content(content="test")

        assert isinstance(result, TextContentBlock)
        assert result.text.value == "test"
        assert result.type == "text"

    def test_string_content_object_have_empty_annotations(self):
        result = to_openai_message_content(content="test")

        assert result.text.annotations == []

    def test_TextContentBlockParam_content_is_converted_to_TextContentBlock(self):
        result = to_openai_message_content(
            content=TextContentBlockParam(text="hello", type="text")
        )

        assert isinstance(result, TextContentBlock)
        assert result.text.value == "hello"
        assert result.type == "text"

    def test_ImageFileContentBlockParam_content_converted_to_ImageFileContentBlock(
        self,
    ):
        result = to_openai_message_content(
            content=ImageFileContentBlockParam(
                image_file=ImageFileParam(file_id="aaa", detail="high"),
                type="image_file",
            )
        )

        assert isinstance(result, ImageFileContentBlock)
        assert result.type == "image_file"
        assert result.image_file.file_id == "aaa"
        assert result.image_file.detail == "high"

    def test_ImageURLContentBlockParam_content_converted_to_ImageURLContentBlock(
        self,
    ):
        result = to_openai_message_content(
            content=ImageURLContentBlockParam(
                image_url=ImageURLParam(url="my-url", detail="low"),
                type="image_url",
            )
        )

        assert isinstance(result, ImageURLContentBlock)
        assert result.type == "image_url"
        assert result.image_url.url == "my-url"
        assert result.image_url.detail == "low"


class TestToOpenaiMessageContentList:
    def test_no_content_is_empty_array(self):
        result = to_openai_message_content_list(content=None)

        assert isinstance(result, list)

    def test_string_content_is_converted_to_TextContentBlock(self):
        result = to_openai_message_content_list(content="test")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].text.value == "test"
        assert result[0].type == "text"


class TestDeserializeMessageContent:
    def test_deserialize_text_content(self):
        result = deserialize_message_content(
            data={"type": "text", "text": {"value": "test", "annotations": []}}
        )

        assert isinstance(result, TextContentBlock)
        assert result.type == "text"
        assert result.text.value == "test"

    def test_deserialize_image_url(self):
        result = deserialize_message_content(
            data={
                "type": "image_url",
                "image_url": {"url": "http://my-image.com", "detail": "low"},
            }
        )

        assert isinstance(result, ImageURLContentBlock)
        assert result.type == "image_url"
        assert result.image_url.url == "http://my-image.com"
        assert result.image_url.detail == "low"

    def test_deserialize_image_file(self):
        result = deserialize_message_content(
            data={
                "type": "image_file",
                "image_file": {"file_id": "abc", "detail": "auto"},
            }
        )

        assert isinstance(result, ImageFileContentBlock)
        assert result.type == "image_file"
        assert result.image_file.file_id == "abc"
        assert result.image_file.detail == "auto"
