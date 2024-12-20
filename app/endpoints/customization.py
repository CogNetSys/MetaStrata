# /app/endpoints/customization.py

from fastapi import APIRouter
import logfire
from app.config import settings
from app.models import PromptSettings

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
    settings.SIMULATION.GRID_SIZE = simulation_settings["grid_size"]
    settings.SIMULATION.NUM_ENTITIES = simulation_settings["num_entities"]
    settings.SIMULATION.MAX_STEPS = simulation_settings["max_steps"]
    settings.SIMULATION.CHEBYSHEV_DISTANCE = simulation_settings["chebyshev_distance"]
    settings.SIMULATION.LLM_MODEL = simulation_settings["llm_model"]
    settings.SIMULATION.LLM_MAX_TOKENS = simulation_settings["llm_max_tokens"]
    settings.SIMULATION.LLM_TEMPERATURE = simulation_settings["llm_temperature"]
    settings.SIMULATION.REQUEST_DELAY = simulation_settings["request_delay"]
    settings.SIMULATION.MAX_CONCURRENT_REQUESTS = simulation_settings["max_concurrent_requests"]

    if settings.LOGFIRE.LOGFIRE_ENABLED:
        logfire.info("Simulation settings updated.")
    return simulation_settings

@router.get("/prompts", tags=["Customization"])
async def get_prompts():
    prompt_settings = {
        "default_message_generation_prompt": settings.SIMULATION.DEFAULT_MESSAGE_GENERATION_PROMPT,
        "default_memory_generation_prompt": settings.SIMULATION.DEFAULT_MEMORY_GENERATION_PROMPT,
        "default_movement_generation_prompt": settings.SIMULATION.DEFAULT_MOVEMENT_GENERATION_PROMPT,
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
