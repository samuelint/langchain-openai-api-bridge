def to_string_content(content) -> str:
    if isinstance(content, str):
        return content

    elif isinstance(content, list):
        if len(content) == 0:
            return ""
        else:
            return content[0]["text"]
