from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "DATAGIT"
    database_url: str = "sqlite:///./datagit.db"

    openai_api_key: str | None = None
    groq_api_key: str

    # OAuth / browser configuration
    frontend_url: str = "http://localhost:5173"
    backend_url: str = "http://127.0.0.1:8000"

    google_client_id: str | None = None
    google_client_secret: str | None = None
    google_redirect_uri: str | None = None

    github_client_id: str | None = None
    github_client_secret: str | None = None
    github_redirect_uri: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
