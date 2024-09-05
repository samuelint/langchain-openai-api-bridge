from typing import AsyncIterator

from pydantic import BaseModel


async def ato_dict(async_iter: AsyncIterator[BaseModel]) -> AsyncIterator[dict]:
    async for obj in async_iter:
        yield obj.dict()
