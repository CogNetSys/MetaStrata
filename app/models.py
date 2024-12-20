import httpx
import logfire
import random
from typing import List

from app.config import Settings, settings, calculate_chebyshev_distance
from app.database import redis, supabase
from pydantic import BaseModel


class Entity(BaseModel):
    id: int
    name: str
    x: int
    y: int
    memory: str = ""  # Default empty memory for new entities


class PromptSettings(BaseModel):
    default_message_generation_prompt: str
    default_memory_generation_prompt: str
    default_movement_generation_prompt: str


class StepRequest(BaseModel):
    steps: int


async def fetch_nearby_messages(entity, entities):
    nearby_entities = [
        a for a in entities if a['id'] != entity['id'] and calculate_chebyshev_distance(entity['x'], entity['y'], a['x'], a['y']) <= settings.SIMULATION.CHEBYSHEV_DISTANCE
    ]
    messages = [await redis.hget(f'entity:{a["id"]}', 'message') for a in nearby_entities]
    return [m for m in messages if m]


# Helper function to fetch Prompts from FastAPI
async def fetch_prompts_from_fastapi():
    async with httpx.AsyncClient() as client:
        response = await client.get('http://localhost:8000/prompts')
        if response.status_code == 200:
            return response.json()  # Return the fetched prompts
        else:
            if settings.LOGFIRE.LOGFIRE_ENABLED:
                logfire.error('Failed to fetch prompts, using default ones.')
            return {}  # Return an empty dict to trigger the default prompts


async def initialize_entities():
    if settings.LOGFIRE.LOGFIRE_ENABLED:
        logfire.info('Resetting simulation state.')
    supabase.table('movements').delete().neq('entity_id', -1).execute()
    supabase.table('entities').delete().neq('id', -1).execute()

    entities = [
        {
            'id': i,
            'name': f'Entity-{i}',
            'x': random.randint(0, settings.SIMULATION.GRID_SIZE - 1),
            'y': random.randint(0, settings.SIMULATION.GRID_SIZE - 1),
            'memory': ''
        }
        for i in range(settings.SIMULATION.NUM_ENTITIES)
    ]
    supabase.table('entities').insert(entities).execute()
    for entity in entities:
        await redis.hset(f'entity:{entity["id"]}', mapping=entity)
    if settings.LOGFIRE.LOGFIRE_ENABLED:
        logfire.info('Entities initialized.')
    return entities
