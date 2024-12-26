# /app/endpoints/customization.py

from fastapi import APIRouter, HTTPException
import logfire
from app.config import settings
from app.utils.models import PromptSettings

router = APIRouter()

@router.get("/settings", tags=["Customization"])
async def get_settings():
    simulation_settings = {
        "grid_size": settings.SIMULATION.GRID_SIZE,
        "num_entities": settings.SIMULATION.NUM_ENTITIES,
        "max_steps": settings.SIMULATION.MAX_STEPS,
        "chebyshev_distance": settings.SIMULATION.CHEBYSHEV_DISTANCE,
        "llm_model": settings.SIMULATION.LLM_MODEL,
        "llm_max_tokens": settings.SIMULATION.LLM_MAX_TOKENS,
        "llm_temperature": settings.SIMULATION.LLM_TEMPERATURE,
        "request_delay": settings.SIMULATION.REQUEST_DELAY,
        "max_concurrent_requests": settings.SIMULATION.MAX_CONCURRENT_REQUESTS,
    }
    if settings.LOGFIRE.LOGFIRE_ENABLED:
        logfire.debug("Simulation settings retrieved.")
    return simulation_settings

@router.post("/settings", tags=["Customization"])
async def set_settings(simulation_settings: dict):
    try:
        # Update settings in the current instance
        for key, value in simulation_settings.items():
            if hasattr(settings.SIMULATION, key.upper()):
                setattr(settings.SIMULATION, key.upper(), value)

        # Save the updated settings to settings.yaml
        settings.SIMULATION.save_settings()

        if settings.LOGFIRE.LOGFIRE_ENABLED:
            logfire.info("Simulation settings updated and saved.")
        return {"message": "Simulation settings updated successfully."}
    except Exception as e:
        if settings.LOGFIRE.LOGFIRE_ENABLED:
            logfire.error(f"Error updating settings: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating settings: {e}")

@router.get("/prompts", tags=["Customization"])
async def get_prompts():
    # Render the default prompts with empty values
    prompt_settings = {
        "default_message_generation_prompt": settings.SIMULATION.DEFAULT_MESSAGE_GENERATION_PROMPT.render(
            entityId="", x=0, y=0, grid_description="", memory="", messages="", distance=0
        ),
        "default_memory_generation_prompt": settings.SIMULATION.DEFAULT_MEMORY_GENERATION_PROMPT.render(
            entityId="", x=0, y=0, grid_description="", memory="", messages=""
        ),
        "default_movement_generation_prompt": settings.SIMULATION.DEFAULT_MOVEMENT_GENERATION_PROMPT.render(
            entityId="", x=0, y=0, grid_description="", memory="", messages=""
        ),
    }
    if settings.LOGFIRE.LOGFIRE_ENABLED:
        logfire.debug("Prompt templates retrieved.")
    return prompt_settings

@router.post("/prompts", tags=["Customization"])
async def set_prompts(prompt_settings: dict):
    settings.SIMULATION.DEFAULT_MESSAGE_GENERATION_PROMPT = prompt_settings["default_message_generation_prompt"]
    settings.SIMULATION.DEFAULT_MEMORY_GENERATION_PROMPT = prompt_settings["default_memory_generation_prompt"]
    settings.SIMULATION.DEFAULT_MOVEMENT_GENERATION_PROMPT = prompt_settings["default_movement_generation_prompt"]

    if settings.LOGFIRE.LOGFIRE_ENABLED:
        logfire.info("Prompt templates updated.")
    return prompt_settings