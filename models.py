from pydantic import BaseModel
from typing import List

class Entity(BaseModel):
    id: int
    name: str
    x: int
    y: int
    memory: str = ""

class SimulationSettings(BaseModel):
    grid_size: int
    num_entities: int
    max_steps: int
    chebyshev_distance: int
    llm_model: str
    llm_max_tokens: int
    llm_temperature: float
    request_delay: float
    max_concurrent_requests: int

class PromptSettings(BaseModel):
    message_generation_prompt: str
    memory_generation_prompt: str
    movement_generation_prompt: str

class BatchMessage(BaseModel):
    entity_id: int
    message: str

class BatchMessagesPayload(BaseModel):
    messages: List[BatchMessage]
