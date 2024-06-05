from typing import Optional
from fastapi import HTTPException


def get_bearer_token(authorization: Optional[str] = None) -> str:
    if authorization and authorization.startswith("Bearer "):
        return authorization[len("Bearer "):]
    raise HTTPException(status_code=401, detail="Invalid authorization header")
