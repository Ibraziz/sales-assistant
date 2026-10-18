from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def discover_root() -> Path:
    return Path.cwd().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)
    project_root: Path = Field(default_factory=discover_root)
    gemini_api_key: SecretStr | None = Field(default=None, repr=False)
    langfuse_public_key: SecretStr | None = Field(default=None, repr=False)
    langfuse_secret_key: SecretStr | None = Field(default=None, repr=False)
    langfuse_base_url: str | None = None
    langfuse_tracing_enabled: bool = True
    langfuse_tracing_environment: str = "development"
    chat_model: str = "gemini-3.5-flash-lite"
    thinking_level: Literal["minimal", "low", "medium", "high"] = "minimal"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = Field(default=768)
    chroma_dir: Path = Path(".local/chroma")
    chroma_product_collection: str
    chroma_service_collection: str
    chroma_memory_collection: str
    data_dir: Path = Path("data")
    retrieval_k: int = Field(default=12)
    history_turns: int = Field(default=6)
    api_timeout_seconds: int = Field(default=45)
    max_api_retries: int = Field(default=2)

    def model_post_init(self, __context):
        self.project_root = self.project_root.resolve()
        self.chroma_dir = (self.project_root / self.chroma_dir).resolve()
        self.data_dir = (self.project_root / self.data_dir).resolve()

    def require_key(self):
        if (
            not self.gemini_api_key
            or not self.gemini_api_key.get_secret_value().strip()
        ):
            raise ValueError("GEMINI_API_KEY is missing. Set it in PROJECT_ROOT/.env.")


def load_settings(root: Path | None = None) -> Settings:
    root = (root or discover_root()).resolve()
    return Settings(
        project_root=root, _env_file=root / ".env", _env_file_encoding="utf-8"
    )
