# /app/utils/config.py:

import os
import yaml
from typing import ClassVar
from pydantic import SecretStr, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from jinja2 import Environment, FileSystemLoader, select_autoescape

# --------------------------------------------------------
# SENSITIVE SETTINGS - Managed via Doppler
# --------------------------------------------------------

class DatabaseSettings(BaseSettings):
    SUPABASE_KEY: SecretStr = Field(..., env="SUPABASE_KEY")
    SUPABASE_URL: str = Field(..., env="SUPABASE_URL")
    use_ssl: bool = Field(False, env="USE_SSL")  # Optional extra field
    ssl_certfile: str = Field(None, env="SSL_CERTFILE")  # Optional extra field
    ssl_keyfile: str = Field(None, env="SSL_KEYFILE")  # Optional extra field

    class Config:
        env_file = ".env"
        extra = "allow"

    @property
    def supabase_key(self) -> str:
        return self.SUPABASE_KEY.get_secret_value()

    @property
    def supabase_url(self) -> str:
        return self.SUPABASE_URL

class GROQSettings(BaseSettings):
    GROQ_API_ENDPOINT: str = Field("https://api.groq.com/openai/v1/chat/completions", env="GROQ_API_ENDPOINT")
    GROQ_API_KEY: SecretStr = Field(..., env="GROQ_API_KEY")  # Required field

    class Config:
        env_file = ".env"
        extra = "allow"  # Allow extra fields

    @property
    def groq_api_endpoint(self) -> str:
        return self.GROQ_API_ENDPOINT

    @property
    def groq_api_key(self) -> str:
        return self.GROQ_API_KEY.get_secret_value()

class RedisSettings(BaseSettings):
    REDIS_ENDPOINT: str = Field(..., env="REDIS_ENDPOINT")  # Required field
    REDIS_PASSWORD: SecretStr = Field(..., env="REDIS_PASSWORD")  # Required field

    class Config:
        env_file = ".env"
        extra = "allow"  # Allow extra fields

    @property
    def redis_endpoint(self) -> str:
        return self.REDIS_ENDPOINT

    @property
    def redis_password(self) -> str:
        return self.REDIS_PASSWORD.get_secret_value()

class AuthSettings(BaseSettings):
    AUTH_TOKEN: SecretStr = Field(..., env="AUTH_TOKEN")  # Required field
    E2B_API_KEY: SecretStr = Field(..., env="E2B_API_KEY")  # Required field

    class Config:
        env_file = ".env"
        extra = "allow"  # Allow extra fields

class LogfireSettings(BaseSettings):
    LOGFIRE_API_KEY: SecretStr = Field(..., env="LOGFIRE_API_KEY")  # Required field
    LOGFIRE_ENDPOINT: str = Field("https://logfire.pydantic.dev", env="LOGFIRE_ENDPOINT")
    LOGFIRE_ENABLED: bool = Field(True, env="LOGFIRE_ENABLED")

    class Config:
        env_file = ".env"
        extra = "allow"  # Allow extra fields

    @property
    def logfire_api_key(self) -> str:
        return self.LOGFIRE_API_KEY.get_secret_value()

    @property
    def logfire_endpoint(self) -> str:
        return self.LOGFIRE_ENDPOINT

    @property
    def logfire_enabled(self) -> bool:
        return self.LOGFIRE_ENABLED

class SimulationSettings(BaseSettings):
    CHEBYSHEV_DISTANCE: float = Field(5.0, env="CHEBYSHEV_DISTANCE")  # Default: 5.0
    GRID_SIZE: int = Field(15, env="GRID_SIZE")  # Default: 15
    LLM_MAX_TOKENS: int = Field(2048, env="LLM_MAX_TOKENS")  # Default: 2048
    LLM_MODEL: str = Field("llama-3.1-8b-instant", env="LLM_MODEL")  # Default value
    LLM_TEMPERATURE: float = Field(0.7, env="LLM_TEMPERATURE")  # Default: 0.7
    LOG_LEVEL: str = Field("DEBUG", env="LOG_LEVEL")  # Default: DEBUG
    MAX_CONCURRENT_REQUESTS: int = Field(5, env="MAX_CONCURRENT_REQUESTS")  # Default: 5
    MAX_STEPS: int = Field(10, env="MAX_STEPS")  # Default: 10
    NUM_ENTITIES: int = Field(3, env="NUM_ENTITIES")  # Default: 3
    REQUEST_DELAY: float = Field(2.3, env="REQUEST_DELAY")  # Default: 2.3

    # Jinja2 environment for prompts - annotated with ClassVar
    prompt_env: ClassVar[Environment] = Environment(
        loader=FileSystemLoader("prompts"),
        autoescape=select_autoescape(['html', 'xml'])
    )

    # Load templates without rendering
    DEFAULT_MESSAGE_GENERATION_PROMPT: ClassVar[Environment.get_template] = prompt_env.get_template("message_generation.jinja")
    DEFAULT_MEMORY_GENERATION_PROMPT: ClassVar[Environment.get_template] = prompt_env.get_template("memory_generation.jinja")
    DEFAULT_MOVEMENT_GENERATION_PROMPT: ClassVar[Environment.get_template] = prompt_env.get_template("movement_generation.jinja")

    # File to store settings
    SETTINGS_FILE: ClassVar[str] = "settings.yaml"

    model_config = SettingsConfigDict(env_file=".env", extra='allow')

    def __init__(self, **data):
        super().__init__(**data)
        self.load_settings()

    def load_settings(self):
        """Loads settings from a YAML file if it exists."""
        if os.path.exists(self.SETTINGS_FILE):
            with open(self.SETTINGS_FILE, "r") as f:
                try:
                    settings_data = yaml.safe_load(f)
                    # Update settings from the YAML file
                    for key, value in settings_data.get("simulation", {}).items():
                        if hasattr(self, key.upper()):
                            setattr(self, key.upper(), value)
                except yaml.YAMLError:
                    print(f"Error: {self.SETTINGS_FILE} is not a valid YAML file. Using default or environment settings.")

    def save_settings(self):
        """Saves the current settings to a YAML file."""
        settings_data = {"simulation": {}}
        for key in self.model_fields.keys():
            # Convert to uppercase for consistency with your code
            upper_key = key.upper()
            if hasattr(self, upper_key):
                # Get the value of the uppercase attribute
                settings_data["simulation"][key] = getattr(self, upper_key)

        with open(self.SETTINGS_FILE, "w") as f:
            yaml.dump(settings_data, f, indent=4)

    # ------------------------------------------------------
    # READ ONLY PROPERTIES - Make these attributes read-only
    # ------------------------------------------------------

    @property
    def chebyshev_distance(self) -> float:
        return self.CHEBYSHEV_DISTANCE

    @property
    def grid_size(self) -> int:
        return self.GRID_SIZE

    @property
    def num_entities(self) -> int:
        return self.NUM_ENTITIES

    @property
    def max_steps(self) -> int:
        return self.MAX_STEPS

    @property
    def grid_description(self) -> str:
        return f"You are in a {self.GRID_SIZE} x {self.GRID_SIZE} field with periodic boundary conditions with {self.NUM_ENTITIES} other lifeforms. You are free to move around the field and interact with other lifeforms."

    @property
    def llm_model(self) -> str:
        return self.LLM_MODEL

    @property
    def llm_max_tokens(self) -> int:
        return self.LLM_MAX_TOKENS

    @property
    def llm_temperature(self) -> float:
        return self.LLM_TEMPERATURE

    @property
    def request_delay(self) -> float:
        return self.REQUEST_DELAY

    @property
    def max_concurrent_requests(self) -> int:
        return self.MAX_CONCURRENT_REQUESTS

    @property
    def log_level(self) -> str:
        return self.LOG_LEVEL
    

class Settings:
    def __init__(self):
        self.DATABASE = DatabaseSettings()
        self.GROQ = GROQSettings()
        self.REDIS = RedisSettings()
        self.AUTH = AuthSettings()
        self.LOGFIRE = LogfireSettings()
        self.SIMULATION = SimulationSettings()

# Global settings instance
settings = Settings()

# Utility function
def calculate_chebyshev_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    """
    Calculate the Chebyshev distance between two points.

    Args:
        x1 (int): X-coordinate of the first point.
        y1 (int): Y-coordinate of the first point.
        x2 (int): X-coordinate of the second point.
        y2 (int): Y-coordinate of the second point.

    Returns:
        int: The Chebyshev distance between two points.
    """
    return max(abs(x1 - x2), abs(y1 - y2))