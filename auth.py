from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

from config import get_settings

API_KEY_HEADER = APIKeyHeader(name="x-api-key", auto_error=False)


async def require_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    settings = get_settings()
    if not api_key or api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key",
        )
    return api_key
