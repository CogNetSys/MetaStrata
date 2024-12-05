from fastapi import APIRouter, HTTPException
from config import SimulationSettings, add_log
from config import (
    GRID_SIZE, NUM_ENTITIES, MAX_STEPS, CHEBYSHEV_DISTANCE,
    LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE,
    REQUEST_DELAY, MAX_CONCURRENT_REQUESTS
)

router = APIRouter()

@router.get("/", response_model=SimulationSettings, tags=["Settings"])
async def get_settings():
    try:
        add_log("Fetching simulation settings.")
        return SimulationSettings(
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
    except Exception as e:
        add_log(f"Error fetching simulation settings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/", response_model=SimulationSettings, tags=["Settings"])
async def set_settings(settings: SimulationSettings):
    try:
        add_log("Updating simulation settings.")
        global GRID_SIZE, NUM_ENTITIES, MAX_STEPS, CHEBYSHEV_DISTANCE
        global LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE, REQUEST_DELAY, MAX_CONCURRENT_REQUESTS

        GRID_SIZE = settings.grid_size
        NUM_ENTITIES = settings.num_entities
        MAX_STEPS = settings.max_steps
        CHEBYSHEV_DISTANCE = settings.chebyshev_distance
        LLM_MODEL = settings.llm_model
        LLM_MAX_TOKENS = settings.llm_max_tokens
        LLM_TEMPERATURE = settings.llm_temperature
        REQUEST_DELAY = settings.request_delay
        MAX_CONCURRENT_REQUESTS = settings.max_concurrent_requests
        return settings
    except Exception as e:
        add_log(f"Error updating simulation settings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
