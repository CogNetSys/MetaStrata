# /app/config.py

import os
from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()

# --------------------------------------------------------
# MODIFIABLE PROPERTIES - These attributes are modifiable
# --------------------------------------------------------
    
class DatabaseSettings:
    SUPABASE_KEY: SecretStr = SecretStr(os.getenv("SUPABASE_KEY"))
    SUPABASE_URL: str = os.getenv("SUPABASE_URL")

class GROQSettings:
    GROQ_API_ENDPOINT: str = os.getenv("GROQ_API_ENDPOINT", "https://api.groq.com/openai/v1/chat/completions")
    GROQ_API_KEY: SecretStr = SecretStr(os.getenv("GROQ_API_KEY"))

class RedisSettings:
    REDIS_ENDPOINT: str = os.getenv("REDIS_ENDPOINT")
    REDIS_PASSWORD: SecretStr = SecretStr(os.getenv("REDIS_PASSWORD"))

class AuthSettings:
    AUTH_TOKEN: SecretStr = SecretStr(os.getenv("AUTH_TOKEN"))
    E2B_API_KEY: SecretStr = SecretStr(os.getenv("E2B_API_KEY"))

class LogfireSettings:
    LOGFIRE_API_KEY: SecretStr = SecretStr(os.getenv("LOGFIRE_API_KEY"))
    LOGFIRE_ENDPOINT: str = os.getenv("LOGFIRE_ENDPOINT", "https://logfire.pydantic.dev")
    LOGFIRE_ENABLED: bool = os.getenv("LOGFIRE_ENABLED", "false").lower() == "true"

class SimulationSettings:
    CHEBYSHEV_DISTANCE: float = float(os.getenv("CHEBYSHEV_DISTANCE", 5.0))  # Default: 5.0
    GRID_SIZE: int = int(os.getenv("GRID_SIZE", 15))  # Default: 15
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", 2048))  # Default: 2048
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")  # Default: "llama-3.3-70b-versatile"
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.7))  # Default: 0.7
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG")  # Default: "DEBUG"
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", 5))  # Default: 5
    MAX_STEPS: int = int(os.getenv("MAX_STEPS", 10))  # Default: 10
    NUM_ENTITIES: int = int(os.getenv("NUM_ENTITIES", 3))  # Default: 3
    REQUEST_DELAY: float = float(os.getenv("REQUEST_DELAY", 2.3))  # Default: 1.2

    DEFAULT_MESSAGE_GENERATION_PROMPT: str = """
    You are lifeform{entityId} at position ({x}, {y}). {grid_description} 
    You have a summary memory of the situation so far: {memory}. 
    You received messages from the surrounding lifeforms: {messages}. 
    Based on the above, you send a message to the surrounding lifeforms. Your message will reach lifeforms up to distance {distance} away. What message do you send?
    Respond with only the message content, and nothing else.
    """

    DEFAULT_MEMORY_GENERATION_PROMPT: str = """
    You are lifeform{entityId} at position ({x}, {y}). {grid_description} 
    You have a summary memory of the situation so far: {memory}. 
    You received messages from the surrounding lifeforms: {messages}. 
    Based on the above, summarize the situation you and the other lifeforms have been in so far for you to remember.
    Respond with only the summary, and nothing else.
    """

    DEFAULT_MOVEMENT_GENERATION_PROMPT: str = """
    You are lifeform{entityId} at position ({x}, {y}). {grid_description} 
    You have a summary memory of the situation so far: {memory}.
    Based on the above, choose your next move. Respond with only one of the following options, and nothing else: "x+1", "x-1", "y+1", "y-1", or "stay".
    Do not provide any explanation or additional text.
    """

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

    @property
    def default_memory_generation_prompt(self) -> str:
        return self.DEFAULT_MEMORY_GENERATION_PROMPT

    @property
    def default_message_generation_prompt(self) -> str:
        return self.DEFAULT_MESSAGE_GENERATION_PROMPT

    @property
    def default_movement_generation_prompt(self) -> str:
        return self.DEFAULT_MOVEMENT_GENERATION_PROMPT

class Settings:
    GROQ: GROQSettings = GROQSettings()
    AUTH: AuthSettings = AuthSettings()
    DATABASE: DatabaseSettings = DatabaseSettings()
    LOGFIRE: LogfireSettings = LogfireSettings()
    REDIS: RedisSettings = RedisSettings()
    SIMULATION: SimulationSettings = SimulationSettings()

# Create a global settings object
settings = Settings()

# Leave this here to prevent circular imports please, I knkow it's not the best place. Whatever.
def calculate_chebyshev_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    """
    Calculate the Chebyshev distance between two points.

    Args:
        x1 (int): X-coordinate of the first point.
        y1 (int): Y-coordinate of the first point.
        x2 (int): X-coordinate of the second point.
        y2 (int): Y-coordinate of the second point.

    Returns:
        int: The Chebyshev distance between the two points.
    """
    return max(abs(x1 - x2), abs(y1 - y2))
