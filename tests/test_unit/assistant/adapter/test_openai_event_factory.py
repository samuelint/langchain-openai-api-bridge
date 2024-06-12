from langchain_openai_api_bridge.assistant.adapter.openai_event_factory import (
    create_langchain_function,
)


class TestCreateLangchainFunction:
    def test_dict_arguments_is_set_to_json(self):
        result = create_langchain_function(arguments={"test": "test"})

        assert result.arguments == '{"test": "test"}'

    def test_dict_output_is_set_to_json(self):
        result = create_langchain_function(output={"test": "test"})

        assert result.output == '{"test": "test"}'

    def test_float_output_is_set_to_string(self):
        result = create_langchain_function(output=2.1)

        assert result.output == "2.1"
