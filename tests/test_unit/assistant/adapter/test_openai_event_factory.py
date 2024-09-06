import json
from langchain_openai_api_bridge.assistant.adapter.openai_event_factory import (
    create_langchain_function,
)
from langchain_core.messages.tool import ToolMessage


class TestCreateLangchainFunction:
    def test_dict_arguments_is_set_to_json(self):
        result = create_langchain_function(arguments={"test": "test"})

        assert result.arguments == '{"test": "test"}'

    def test_dict_output_is_set_to_json(self):
        result = create_langchain_function(
            arguments={"a": "a"}, output={"test": "test"}
        )

        assert result.output == '{"test": "test"}'

    def test_float_output_is_set_to_string(self):
        result = create_langchain_function(arguments={"a": "a"}, output=2.1)

        assert result.output == "2.1"

    def test_ToolMessageOutput_is_serialized_to_json(self):
        tool_message_output = ToolMessage(
            content="some",
            tool_call_id="123",
        )
        result = create_langchain_function(
            arguments={"a": "a"}, output=tool_message_output
        )

        output = json.loads(result.output)
        assert output["content"] == "some"
        assert output["tool_call_id"] == "123"
        assert output["status"] == "success"
        assert output.get("artifact") is None

    def test_ToolMessageOutput_with_artifact_is_serialized_to_json(self):
        tool_message_output = ToolMessage(
            content="some",
            tool_call_id="123",
            artifact={"test": "test"},
        )
        result = create_langchain_function(
            arguments={"a": "a"}, output=tool_message_output
        )

        output = json.loads(result.output)
        assert output["artifact"] == {"test": "test"}
