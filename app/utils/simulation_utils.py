# /app/utils/simulation_utils.py

from pydantic import BaseModel, Field, field_validator

class MessageResponse(BaseModel):
    message: str = Field(..., description="The generated message from the entity.")

    @field_validator('message')
    def check_message(cls, v):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Message must be a non-empty string.")
        return v

class MemoryResponse(BaseModel):
    memory: str = Field(..., description="The updated memory of the entity.")

    @field_validator('memory')
    def check_memory(cls, v):
        if not isinstance(v, str):
            raise ValueError("Memory must be a string.")
        return v

class MovementResponse(BaseModel):
    movement: str = Field(..., description="The movement command chosen by the entity.")

    @field_validator('movement')
    def check_movement(cls, v):
        allowed_movements = ["x+1", "x-1", "y+1", "y-1", "stay"]
        if v.lower() not in allowed_movements:
            raise ValueError(f"Invalid movement command. Allowed commands are: {allowed_movements}")
        return v.lower()

class LLMRequestError(BaseModel):
    error: str = Field(..., description="Error message in case of LLM request failure.")

    @field_validator('error')
    def check_error(cls, v):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Error message must be a non-empty string.")
        return v