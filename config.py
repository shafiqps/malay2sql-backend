from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    database_url: str
    model_path: str
    reranker_path: str

    class Config:
        env_file = ".env" 