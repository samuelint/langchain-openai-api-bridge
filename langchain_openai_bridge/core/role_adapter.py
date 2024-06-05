def to_openai_role(role: str) -> str:
    match role:
        case "ai":
            return "assistant"
        case _:
            return role
