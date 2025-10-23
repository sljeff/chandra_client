from dotenv import find_dotenv
from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IMAGE_DPI: int = 200
    MIN_IMAGE_DIM: int = 1024
    MAX_OUTPUT_TOKENS: int = 8192

    # vLLM server settings
    VLLM_API_KEY: str = ""
    VLLM_API_BASE: str = "http://localhost:8000/v1"
    VLLM_MODEL_NAME: str = "chandra"
    MAX_VLLM_RETRIES: int = 6

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


settings = Settings()
