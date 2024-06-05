from fastapi import HTTPException
import pytest
from langchain_openai_bridge.fastapi.token_getter import get_bearer_token


class TestGetBearerToken:
    def test_extract_bearer_token_from_authorization_header(self):
        result = get_bearer_token("Bearer some-token")

        assert result == "some-token"

    def test_exception_is_raised_when_Bearer_is_not_present(self):
        with pytest.raises(HTTPException) as e:
            get_bearer_token("some-token")
        assert e.value.status_code == 401

    def test_exception_is_raised_when_authorization_header_is_missing(self):
        with pytest.raises(HTTPException) as e:
            get_bearer_token()
        assert e.value.status_code == 401
