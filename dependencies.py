from functools import lru_cache
from services.malay2sql_service import Malay2SQLService
from config import Settings
import redis

@lru_cache()
def get_settings():
    return Settings()

@lru_cache()
def get_redis_client():
    settings = get_settings()
    return redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db
    )

@lru_cache()
def get_malay2sql_service() -> Malay2SQLService:
    settings = get_settings()
    redis_client = get_redis_client()
    return Malay2SQLService(
        openai_api_key=settings.openai_api_key,
        cache_client=redis_client
    ) 