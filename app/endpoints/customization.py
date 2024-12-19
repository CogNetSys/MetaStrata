# app/endpoints/customization.py

from fastapi import APIRouter
import logfire

# Import shared models and constants
from app.models import SimulationSettings, PromptSettings, LOGFIRE_ENABLED
from app.main import GRID_SIZE, NUM_ENTITIES, MAX_STEPS, CHEBYSHEV_DISTANCE, LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE, REQUEST_DELAY, MAX_CONCURRENT_REQUESTS, DEFAULT_MEMORY_GENERATION_PROMPT, DEFAULT_MESSAGE_GENERATION_PROMPT, DEFAULT_MOVEMENT_GENERATION_PROMPT

router = APIRouter()

@router.get("/settings", response_model=SimulationSettings, tags=["Customization"])
async def get_settings():
    settings = SimulationSettings(
        grid_size=GRID_SIZE,
        num_entities=NUM_ENTITIES,
        max_steps=MAX_STEPS,
        chebyshev_distance=CHEBYSHEV_DISTANCE,
        llm_model=LLM_MODEL,
        llm_max_tokens=LLM_MAX_TOKENS,
        llm_temperature=LLM_TEMPERATURE,
        request_delay=REQUEST_DELAY,
        max_concurrent_requests=MAX_CONCURRENT_REQUESTS
    )
    if LOGFIRE_ENABLED:
        logfire.debug("Simulation settings retrieved.")
    return settings

@router.post("/settings", response_model=SimulationSettings, tags=["Customization"])
async def set_settings(settings: SimulationSettings):
    global GRID_SIZE, NUM_ENTITIES, MAX_STEPS, CHEBYSHEV_DISTANCE, LLM_MODEL
    global LLM_MAX_TOKENS, LLM_TEMPERATURE, REQUEST_DELAY, MAX_CONCURRENT_REQUESTS

    GRID_SIZE = settings.grid_size
    NUM_ENTITIES = settings.num_entities
    MAX_STEPS = settings.max_steps
    CHEBYSHEV_DISTANCE = settings.chebyshev_distance
    LLM_MODEL = settings.llm_model
    LLM_MAX_TOKENS = settings.llm_max_tokens
    LLM_TEMPERATURE = settings.llm_temperature
    REQUEST_DELAY = settings.request_delay
    MAX_CONCURRENT_REQUESTS = settings.max_concurrent_requests

    if LOGFIRE_ENABLED:
        logfire.info("Simulation settings updated.")
    return settings

@router.get("/prompts", response_model=PromptSettings, tags=["Customization"])
async def get_prompts():
    prompts = PromptSettings(
        message_generation_prompt=DEFAULT_MESSAGE_GENERATION_PROMPT,
        memory_generation_prompt=DEFAULT_MEMORY_GENERATION_PROMPT,
        movement_generation_prompt=DEFAULT_MOVEMENT_GENERATION_PROMPT
    )
    if LOGFIRE_ENABLED:
        logfire.debug("Prompt templates retrieved.")
    return prompts

@router.post("/prompts", response_model=PromptSettings, tags=["Customization"])
async def set_prompts(prompts: PromptSettings):
    global DEFAULT_MESSAGE_GENERATION_PROMPT, DEFAULT_MEMORY_GENERATION_PROMPT, DEFAULT_MOVEMENT_GENERATION_PROMPT

    DEFAULT_MESSAGE_GENERATION_PROMPT = prompts.message_generation_prompt
    DEFAULT_MEMORY_GENERATION_PROMPT = prompts.memory_generation_prompt
    DEFAULT_MOVEMENT_GENERATION_PROMPT = prompts.movement_generation_prompt

    if LOGFIRE_ENABLED:
        logfire.info("Prompt templates updated.")
    return prompts
