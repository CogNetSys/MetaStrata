import asyncio
import logging
from fastapi import APIRouter, HTTPException
from typing import List
from fastapi.responses import JSONResponse
import httpx
from pydantic import BaseModel

# Assuming redis, logger, and other necessary components are imported from main.py
from app.main import StepRequest, chebyshev_distance, redis, logger, supabase, initialize_entities, fetch_prompts_from_fastapi, fetch_nearby_messages

from app.main import (
    GROQ_API_KEY,
    GROQ_API_ENDPOINT,
    GRID_SIZE,
    NUM_ENTITIES,
    MAX_STEPS,
    CHEBYSHEV_DISTANCE,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    REQUEST_DELAY,
    MAX_CONCURRENT_REQUESTS,
    GRID_DESCRIPTION,
    DEFAULT_MESSAGE_GENERATION_PROMPT,
    DEFAULT_MEMORY_GENERATION_PROMPT,
    DEFAULT_MOVEMENT_GENERATION_PROMPT,
    logger,
)

logger = logging.getLogger("websocket")
router = APIRouter()

# Semaphore for throttling concurrent requests
global_request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Entity Model (Pydantic)
class Entity(BaseModel):
    id: int
    name: str
    x: int
    y: int
    memory: str = ""  # Default empty memory for new entities

# Simulation Settings Model
class SimulationSettings(BaseModel):
    grid_size: int = 30
    num_entities: int = 10
    max_steps: int = 100
    chebyshev_distance: int = 5
    llm_model: str = "llama-3.2-11b-vision-preview"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.7
    request_delay: float = 2.2
    max_concurrent_requests: int = 1

# Prompt Settings Model
class PromptSettings(BaseModel):
    message_generation_prompt: str
    memory_generation_prompt: str
    movement_generation_prompt: str

# Helper Functions

def construct_prompt(template, entity, messages):
    messages_str = "\n".join(messages) if messages else "No recent messages."
    memory = entity.get("memory", "No prior memory.")
    return template.format(
        entityId=entity["id"], x=entity["x"], y=entity["y"],
        grid_description=GRID_DESCRIPTION, memory=memory,
        messages=messages_str, distance=CHEBYSHEV_DISTANCE
    )

async def send_llm_request(prompt, max_retries=3, base_delay=2):
    global stop_signal
    async with global_request_semaphore:
        if stop_signal:
            logger.info("Stopping LLM request due to stop signal.")
            return {"message": "", "memory": "", "movement": "stay"}

        headers = {
            'Authorization': f'Bearer {GROQ_API_KEY}',
            'Content-Type': 'application/json'
        }
        body = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": LLM_TEMPERATURE
        }

        attempt = 0
        while attempt <= max_retries:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(GROQ_API_ENDPOINT, headers=headers, json=body)
                    if response.status_code == 429:
                        attempt += 1
                        if attempt > max_retries:
                            logger.error("Exceeded max retries for LLM request.")
                            return {"message": "", "memory": "", "movement": "stay"}
                        delay = base_delay * (2 ** (attempt - 1))
                        logger.warning(f"Received 429 Too Many Requests. Retrying in {delay} seconds...")
                        await asyncio.sleep(delay)
                        continue
                    response.raise_for_status()
                    result = response.json()

                    # Validate expected keys
                    if not all(key in result for key in ["choices"]):
                        logger.warning(f"Incomplete response from LLM: {result}")
                        return {"message": "", "memory": "", "movement": "stay"}

                    # Extract content from choices
                    content = result["choices"][0]["message"]["content"].strip()

                    # Depending on the prompt, categorize the response
                    if "What message do you send?" in prompt:
                        await asyncio.sleep(REQUEST_DELAY)
                        return {"message": content}
                    elif "summarize the situation" in prompt:
                        await asyncio.sleep(REQUEST_DELAY)
                        return {"memory": content}
                    elif "choose your next move" in prompt:
                        # Extract movement command
                        valid_commands = ["x+1", "x-1", "y+1", "y-1", "stay"]
                        content_lower = content.lower()
                        for cmd in valid_commands:
                            if cmd == content_lower:
                                movement = cmd
                                break
                        else:
                            logger.warning(f"Invalid movement command in LLM response: {content}")
                            movement = "stay"  # Default to "stay" if invalid
                        await asyncio.sleep(REQUEST_DELAY)
                        return {"movement": movement}
                    else:
                        logger.warning(f"Unexpected prompt type: {prompt}")
                        await asyncio.sleep(REQUEST_DELAY)
                        return {"message": "", "memory": "", "movement": "stay"}
            except Exception as e:
                logger.error(f"Error during LLM request: {e}")
                return {"message": "", "memory": "", "movement": "stay"}

# Simulation API Endpoints
@router.post("/reset")
async def reset_simulation():
    global stop_signal
    stop_signal = False  # Reset stop signal before starting
    await redis.flushdb()
    entities = await initialize_entities()
    return JSONResponse({"status": "Simulation reset successfully.", "entities": entities})

@router.post("/start")
async def start_simulation():
    global stop_signal
    stop_signal = False  # Reset stop signal before starting
    entities = await initialize_entities()
    return JSONResponse({"status": "Simulation started successfully.", "entities": entities})

@router.post("/step")
async def perform_steps(request: StepRequest):
    global stop_signal
    stop_signal = False  # Reset stop signal before starting steps

    # Fetch the current prompt templates from FastAPI
    prompts = await fetch_prompts_from_fastapi()

    entities = [
        {
            "id": int(entity["id"]),
            "name": entity["name"],
            "x": int(entity["x"]),
            "y": int(entity["y"]),
            "memory": entity.get("memory", "")
        }
        for entity in [await redis.hgetall(f"entity:{i}") for i in range(NUM_ENTITIES)]
    ]

    for _ in range(request.steps):
        if stop_signal:
            logger.info("Stopping steps due to stop signal.")
            break

        # Use either custom prompts from FastAPI or fall back to default prompts
        message_prompt = prompts.get("message_generation_prompt", DEFAULT_MESSAGE_GENERATION_PROMPT)
        memory_prompt = prompts.get("memory_generation_prompt", DEFAULT_MEMORY_GENERATION_PROMPT)
        movement_prompt = prompts.get("movement_generation_prompt", DEFAULT_MOVEMENT_GENERATION_PROMPT)

        # Message Generation
        for entity in entities:
            message_result = await send_llm_request(
                message_prompt.format(
                    entityId=entity["id"], x=entity["x"], y=entity["y"],
                    grid_description=GRID_DESCRIPTION,
                    memory=entity["memory"], messages="".join(await fetch_nearby_messages(entity, entities)),
                    distance=CHEBYSHEV_DISTANCE
                )
            )
            if "message" not in message_result:
                logger.warning(f"Skipping message update for Entity {entity['id']} due to invalid response.")
                continue
            await redis.hset(f"entity:{entity['id']}", "message", message_result.get("message", ""))
            await asyncio.sleep(REQUEST_DELAY)

        if stop_signal:
            logger.info("Stopping after message generation due to stop signal.")
            break

        # Memory Generation
        for entity in entities:
            memory_result = await send_llm_request(
                memory_prompt.format(
                    entityId=entity["id"], x=entity["x"], y=entity["y"],
                    grid_description=GRID_DESCRIPTION,
                    memory=entity["memory"], messages="".join(await fetch_nearby_messages(entity, entities)),
                    distance=CHEBYSHEV_DISTANCE
                )
            )
            if "memory" not in memory_result:
                logger.warning(f"Skipping memory update for Entity {entity['id']} due to invalid response.")
                continue
            await redis.hset(f"entity:{entity['id']}", "memory", memory_result.get("memory", entity["memory"]))
            await asyncio.sleep(REQUEST_DELAY)

        if stop_signal:
            logger.info("Stopping after memory generation due to stop signal.")
            break

        # Movement Generation
        for entity in entities:
            movement_result = await send_llm_request(
                movement_prompt.format(
                    entityId=entity["id"], x=entity["x"], y=entity["y"],
                    grid_description=GRID_DESCRIPTION,
                    memory=entity["memory"], messages="".join(await fetch_nearby_messages(entity, entities)),
                    distance=CHEBYSHEV_DISTANCE
                )
            )
            if "movement" not in movement_result:
                logger.warning(f"Skipping movement update for Entity {entity['id']} due to invalid response.")
                continue

            # Apply movement logic
            movement = movement_result.get("movement", "stay").strip().lower()
            initial_position = (entity["x"], entity["y"])

            if movement == "x+1":
                entity["x"] = (entity["x"] + 1) % GRID_SIZE
            elif movement == "x-1":
                entity["x"] = (entity["x"] - 1) % GRID_SIZE
            elif movement == "y+1":
                entity["y"] = (entity["y"] + 1) % GRID_SIZE
            elif movement == "y-1":
                entity["y"] = (entity["y"] - 1) % GRID_SIZE
            elif movement == "stay":
                logger.info(f"Entity {entity['id']} stays in place at {initial_position}.")
                continue  # No update needed for stay
            else:
                logger.warning(f"Invalid movement command for Entity {entity['id']}: {movement}")
                continue  # Skip invalid commands

            # Log and update position
            logger.info(f"Entity {entity['id']} moved from {initial_position} to ({entity['x']}, {entity['y']}) with action '{movement}'.")
            await redis.hset(f"entity:{entity['id']}", mapping={"x": entity["x"], "y": entity["y"]})
            await asyncio.sleep(REQUEST_DELAY)

    return JSONResponse({"status": f"Performed {request.steps} step(s)."})

@router.post("/stop")
async def stop_simulation():
    global stop_signal
    stop_signal = True
    logger.info("Stop signal triggered.")
    return JSONResponse({"status": "Simulation stopping."})

@router.post("/entities", response_model=Entity)
async def create_entity(entity: Entity):
    # Create entity data in Redis
    await redis.hset(f"entity:{entity.id}", mroutering=entity.dict())
    
    # Optionally, store entity in Supabase for persistent storage
    supabase.table("entities").insert(entity.dict()).execute()
    
    return entity

@router.get("/entities/{entity_id}", response_model=Entity)
async def get_entity(entity_id: int):
    # Fetch entity data from Redis
    entity_data = await redis.hgetall(f"entity:{entity_id}")
    if not entity_data:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    # Convert Redis data into an Entity model (ensure proper types are cast)
    entity = Entity(
        id=entity_id,
        name=entity_data.get("name"),
        x=int(entity_data.get("x")),
        y=int(entity_data.get("y")),
        memory=entity_data.get("memory", "")
    )
    return entity

@router.put("/entities/{entity_id}", response_model=Entity)
async def update_entity(entity_id: int, entity: Entity):
    # Update entity data in Redis (assuming 'hset' is used to update fields)
    await redis.hset(f"entity:{entity_id}", mroutering=entity.dict())
    
    # Optionally, update entity in Supabase for persistent storage
    supabase.table("entities").update(entity.dict()).eq("id", entity_id).execute()
    
    return entity

@router.post("/entities/{entity_id}/send_message")
async def send_message(entity_id: int, message: str):
    # Get the entity's current position and details
    entity_data = await redis.hgetall(f"entity:{entity_id}")
    if not entity_data:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    # Save the message in Redis for the entity
    await redis.hset(f"entity:{entity_id}", "message", message)
    
    return {"status": "Message sent successfully", "message": message}

@router.delete("/entities/{entity_id}")
async def delete_entity(entity_id: int):
    # Delete entity from Redis
    await redis.delete(f"entity:{entity_id}")
    
    # Optionally, delete entity from Supabase
    supabase.table("entities").delete().eq("id", entity_id).execute()
    
    return {"status": "Entity deleted successfully"}

@router.get("/entities/{entity_id}/nearby", response_model=List[Entity])
async def get_nearby_entities(entity_id: int):
    # Get the entity's position from Redis
    entity_data = await redis.hgetall(f"entity:{entity_id}")
    if not entity_data:
        raise HTTPException(status_code=404, detail="Entity not found")

    entity = Entity(**entity_data)
    
    # Fetch all entities except the current one
    all_entities = []
    for i in range(NUM_ENTITIES):
        if i != entity_id:
            entity_info = await redis.hgetall(f"entity:{i}")
            if entity_info and all(key in entity_info for key in ["id", "name", "x", "y"]):
                all_entities.routerend(Entity(**entity_info))
            else:
                logger.warning(f"Missing or incomplete data for entity {i}. Skipping.")
    
    # Filter nearby entities based on Chebyshev distance
    nearby_entities = [
        a for a in all_entities
        if chebyshev_distance(entity.x, entity.y, a.x, a.y) <= CHEBYSHEV_DISTANCE
    ]
    
    return nearby_entities

@router.post("/sync_entities")
async def sync_entities():
    all_entities = []
    for i in range(NUM_ENTITIES):
        entity_data = await redis.hgetall(f"entity:{i}")
        if entity_data and all(key in entity_data for key in ["id", "name", "x", "y"]):
            all_entities.routerend(Entity(**entity_data))
        else:
            logger.warning(f"Missing or incomplete data for entity {i}. Skipping.")
    
    for entity in all_entities:
        supabase.table("entities").upsert(entity.dict()).execute()
    
    return {"status": "Entities synchronized between Redis and Supabase"}

@router.get("/settings", response_model=SimulationSettings)
async def get_settings():
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

@router.post("/settings", response_model=SimulationSettings)
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

    return settings

# Endpoint to get current prompt templates
@router.get("/prompts", response_model=PromptSettings)
async def get_prompts():
    return PromptSettings(
        message_generation_prompt=DEFAULT_MESSAGE_GENERATION_PROMPT,
        memory_generation_prompt=DEFAULT_MEMORY_GENERATION_PROMPT,
        movement_generation_prompt=DEFAULT_MOVEMENT_GENERATION_PROMPT
    )

# Endpoint to set new prompt templates
@router.post("/prompts", response_model=PromptSettings)
async def set_prompts(prompts: PromptSettings):
    global DEFAULT_MESSAGE_GENERATION_PROMPT, DEFAULT_MEMORY_GENERATION_PROMPT, DEFAULT_MOVEMENT_GENERATION_PROMPT

    DEFAULT_MESSAGE_GENERATION_PROMPT = prompts.message_generation_prompt
    DEFAULT_MEMORY_GENERATION_PROMPT = prompts.memory_generation_prompt
    DEFAULT_MOVEMENT_GENERATION_PROMPT = prompts.movement_generation_prompt

    return prompts