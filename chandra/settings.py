from dotenv import find_dotenv
from pydantic import computed_field
from pydantic_settings import BaseSettings
import torch
import os


class Settings(BaseSettings):
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IMAGE_DPI: int = 96
    MODEL_CHECKPOINT: str = "datalab-to/chandra-0.2.1"
    TORCH_DEVICE: str | None = None
    MAX_OUTPUT_TOKENS: int = 2048

    @computed_field
    @property
    def TORCH_DEVICE_MODEL(self) -> str:
        if self.TORCH_DEVICE is not None:
            return self.TORCH_DEVICE

        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    @computed_field
    @property
    def TORCH_DTYPE(self) -> torch.dtype:
        return torch.bfloat16

    @computed_field
    @property
    def TORCH_ATTN_IMPLEMENTATION(self) -> str:
        if self.TORCH_DEVICE_MODEL == "cuda":
            return "flash_attention_2"
        else:
            return "sdpa"

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


settings = Settings()
