# app/models.py

from pydantic import BaseModel

# Simulation settings model
class SimulationSettings(BaseModel):
    grid_size: int
    num_entities: int
    max_steps: int
    chebyshev_distance: float
    llm_model: str
    llm_max_tokens: int
    llm_temperature: float
    request_delay: float
    max_concurrent_requests: int

# Prompt settings model
class PromptSettings(BaseModel):
    message_generation_prompt: str
    memory_generation_prompt: str
    movement_generation_prompt: str