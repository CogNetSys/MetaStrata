import asyncio
import logging
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from typing import List
from fastapi.responses import JSONResponse
import httpx
import logfire
from pydantic import BaseModel
from app.main import (
    StepRequest, chebyshev_distance, redis, logger, supabase, initialize_entities,
    fetch_prompts_from_fastapi, fetch_nearby_messages, connected_clients, GROQ_API_KEY, GROQ_API_ENDPOINT,
    GRID_SIZE, NUM_ENTITIES, MAX_STEPS, CHEBYSHEV_DISTANCE, LLM_MODEL,
    LLM_MAX_TOKENS, LLM_TEMPERATURE, REQUEST_DELAY, MAX_CONCURRENT_REQUESTS,
    GRID_DESCRIPTION, DEFAULT_MESSAGE_GENERATION_PROMPT,
    DEFAULT_MEMORY_GENERATION_PROMPT, DEFAULT_MOVEMENT_GENERATION_PROMPT,
    send_log_message
)

# Global variable to track connected WebSocket clients
connected_clients: List[WebSocket] = []

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
    grid_size: int = GRID_SIZE
    num_entities: int = NUM_ENTITIES
    max_steps: int = MAX_STEPS
    chebyshev_distance: int = CHEBYSHEV_DISTANCE
    llm_model: str = LLM_MODEL
    llm_max_tokens: int = LLM_MAX_TOKENS
    llm_temperature: float = LLM_TEMPERATURE
    request_delay: float = REQUEST_DELAY
    max_concurrent_requests: int = MAX_CONCURRENT_REQUESTS

# Prompt Settings Model
class PromptSettings(BaseModel):
    message_generation_prompt: str
    memory_generation_prompt: str
    movement_generation_prompt: str

# Helper Functions

def get_simulation_settings() -> SimulationSettings:
    return SimulationSettings()

def construct_prompt(template, entity, messages):
    messages_str = "\n".join(messages) if messages else "No recent messages."
    memory = entity.get("memory", "No prior memory.")
    return template.format(
        entityId=entity["id"], x=entity["x"], y=entity["y"],
        grid_description=GRID_DESCRIPTION, memory=memory,
        messages=messages_str, distance=CHEBYSHEV_DISTANCE
    )

# Configure Python's logging
logging.basicConfig(level=logging.DEBUG)

# Save the original logfire.debug method
original_debug = logfire.debug

def patched_debug(message, *args, **kwargs):
    """
    Redirect logfire.debug calls to Python's logging.debug.
    """
    if args:
        message = message.format(*args)
    logging.debug(message)  # Send to Python logging instead of Logfire
    # Optionally, send to Logfire for non-debug levels
    # original_debug(message, **kwargs)

# Patch logfire.debug
logfire.debug = patched_debug

async def send_llm_request(prompt, max_retries=3, base_delay=2):
    global stop_signal
    async with global_request_semaphore:
        if stop_signal:
            logfire.info("Stopping LLM request due to stop signal.")
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
                            logfire.error("Exceeded max retries for LLM request.")
                            return {"message": "", "memory": "", "movement": "stay"}
                        delay = base_delay * (2 ** (attempt - 1))
                        logfire.warning(f"Received 429 Too Many Requests. Retrying in {delay} seconds...")
                        await asyncio.sleep(delay)
                        continue
                    response.raise_for_status()
                    result = response.json()

                    # Validate expected keys
                    if not all(key in result for key in ["choices"]):
                        logfire.warning(f"Incomplete response from LLM: {result}")
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
                            logfire.warning(f"Invalid movement command in LLM response: {content}")
                            movement = "stay"  # Default to "stay" if invalid
                        await asyncio.sleep(REQUEST_DELAY)
                        return {"movement": movement}
                    else:
                        logfire.warning(f"Unexpected prompt type: {prompt}")
                        await asyncio.sleep(REQUEST_DELAY)
                        return {"message": "", "memory": "", "movement": "stay"}
            except Exception as e:
                logfire.error(f"Error during LLM request: {e}")
                return {"message": "", "memory": "", "movement": "stay"}

# Simulation API Endpoints

@router.post("/reset_and_initialize")
async def reset_and_initialize():
    global stop_signal
    stop_signal = False  # Reset stop signal before starting
    await redis.flushdb()
    entities = await initialize_entities()
    logfire.info("Simulation reset and initialized successfully.")
    return JSONResponse({"status": "Simulation reset and initialized successfully.", "entities": entities})

@router.post("/step")
async def perform_steps(request: StepRequest):
    global stop_signal
    stop_signal = False  # Reset stop signal before starting steps

    logfire.info(f"Starting {request.steps} step(s).")

    # Fetch the current prompt templates from FastAPI
    prompts = await fetch_prompts_from_fastapi()
    logfire.debug(f"Fetched prompts: {prompts}")

    entities = []
    for i in range(NUM_ENTITIES):
        entity_data = await redis.hgetall(f"entity:{i}")
        if entity_data:
            entities.append({
                "id": int(entity_data["id"]),
                "name": entity_data["name"],
                "x": int(entity_data["x"]),
                "y": int(entity_data["y"]),
                "memory": entity_data.get("memory", "")
            })

    logfire.info(f"Loaded {len(entities)} entities for simulation.")

    for _ in range(request.steps):
        if stop_signal:
            logfire.info("Stopping steps due to stop signal.")
            break

        # Use either custom prompts from FastAPI or fall back to default prompts
        message_prompt = prompts.get("message_generation_prompt", DEFAULT_MESSAGE_GENERATION_PROMPT)
        memory_prompt = prompts.get("memory_generation_prompt", DEFAULT_MEMORY_GENERATION_PROMPT)
        movement_prompt = prompts.get("movement_generation_prompt", DEFAULT_MOVEMENT_GENERATION_PROMPT)

        # Message Generation
        for entity in entities:
            if stop_signal:
                logfire.info("Stopping after message generation due to stop signal.")
                break
            nearby_messages = await fetch_nearby_messages(entity, entities)
            logfire.debug(f"Entity {entity['id']} nearby messages: {nearby_messages}")

            message_result = await send_llm_request(
                message_prompt.format(
                    entityId=entity["id"], x=entity["x"], y=entity["y"],
                    grid_description=GRID_DESCRIPTION,
                    memory=entity["memory"], messages="\n".join(nearby_messages),
                    distance=CHEBYSHEV_DISTANCE
                )
            )
            logfire.debug(f"LLM message result for Entity {entity['id']}: {message_result}")

            if "message" not in message_result:
                logfire.warning(f"Skipping message update for Entity {entity['id']} due to invalid response.")
                continue
            await redis.hset(f"entity:{entity['id']}", "message", message_result.get("message", ""))
            logfire.info(f"Entity {entity['id']} message updated to: {message_result.get('message')}")
            await asyncio.sleep(REQUEST_DELAY)

        if stop_signal:
            logfire.info("Stopping after message generation due to stop signal.")
            break

        # Memory Generation
        for entity in entities:
            if stop_signal:
                logfire.info("Stopping after memory generation due to stop signal.")
                break
            nearby_messages = await fetch_nearby_messages(entity, entities)
            logfire.debug(f"Entity {entity['id']} nearby messages for memory: {nearby_messages}")

            memory_result = await send_llm_request(
                memory_prompt.format(
                    entityId=entity["id"], x=entity["x"], y=entity["y"],
                    grid_description=GRID_DESCRIPTION,
                    memory=entity["memory"], messages="\n".join(nearby_messages),
                    distance=CHEBYSHEV_DISTANCE
                )
            )
            logfire.debug(f"LLM memory result for Entity {entity['id']}: {memory_result}")

            if "memory" not in memory_result:
                logfire.warning(f"Skipping memory update for Entity {entity['id']} due to invalid response.")
                continue
            updated_memory = memory_result.get("memory", entity["memory"])
            await redis.hset(f"entity:{entity['id']}", "memory", updated_memory)
            logfire.info(f"Entity {entity['id']} memory updated to: {updated_memory}")
            await asyncio.sleep(REQUEST_DELAY)

        if stop_signal:
            logfire.info("Stopping after memory generation due to stop signal.")
            break

        # Movement Generation
        for entity in entities:
            if stop_signal:
                logfire.info("Stopping after movement generation due to stop signal.")
                break
            nearby_messages = await fetch_nearby_messages(entity, entities)
            logfire.debug(f"Entity {entity['id']} nearby messages for movement: {nearby_messages}")

            movement_result = await send_llm_request(
                movement_prompt.format(
                    entityId=entity["id"], x=entity["x"], y=entity["y"],
                    grid_description=GRID_DESCRIPTION,
                    memory=entity["memory"], messages="\n".join(nearby_messages),
                    distance=CHEBYSHEV_DISTANCE
                )
            )
            logfire.debug(f"LLM movement result for Entity {entity['id']}: {movement_result}")

            if "movement" not in movement_result:
                logfire.warning(f"Skipping movement update for Entity {entity['id']} due to invalid response.")
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
                logfire.info(f"Entity {entity['id']} stays in place at {initial_position}.")
                continue  # No update needed for stay
            else:
                logfire.warning(f"Invalid movement command for Entity {entity['id']}: {movement}")
                continue  # Skip invalid commands

            # Log and update position
            logfire.info(f"Entity {entity['id']} moved from {initial_position} to ({entity['x']}, {entity['y']}) with action '{movement}'.")
            await redis.hset(f"entity:{entity['id']}", mapping={"x": entity["x"], "y": entity["y"]})
            logfire.debug(f"Entity {entity['id']} position updated in Redis.")
            await asyncio.sleep(REQUEST_DELAY)

    logfire.info(f"Completed {request.steps} step(s).")
    return JSONResponse({"status": f"Performed {request.steps} step(s)."})

@router.post("/stop")
async def stop_simulation():
    global stop_signal
    stop_signal = True
    logfire.info("Stop signal triggered.")
    return JSONResponse({"status": "Simulation stopping."})

@router.post("/entities", response_model=Entity)
async def create_entity(entity: Entity):
    # Create entity data in Redis
    await redis.hset(f"entity:{entity.id}", mapping=entity.dict())
    logfire.info(f"Entity {entity.id} created in Redis.")

    # Optionally, store entity in Supabase for persistent storage
    supabase.table("entities").insert(entity.dict()).execute()
    logfire.info(f"Entity {entity.id} stored in Supabase.")
    
    return entity

@router.get("/entities/{entity_id}", response_model=Entity)
async def get_entity(entity_id: int):
    # Fetch entity data from Redis
    entity_data = await redis.hgetall(f"entity:{entity_id}")
    if not entity_data:
        logfire.error(f"Entity {entity_id} not found.")
        raise HTTPException(status_code=404, detail="Entity not found")
    
    # Convert Redis data into an Entity model (ensure proper types are cast)
    try:
        entity = Entity(
            id=int(entity_data["id"]),
            name=entity_data["name"],
            x=int(entity_data["x"]),
            y=int(entity_data["y"]),
            memory=entity_data.get("memory", "")
        )
        logfire.debug(f"Entity {entity_id} retrieved from Redis.")
        return entity
    except (KeyError, ValueError) as e:
        logfire.error(f"Error parsing entity {entity_id} data: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving entity data")

@router.put("/entities/{entity_id}", response_model=Entity)
async def update_entity(entity_id: int, entity: Entity):
    # Update entity data in Redis
    await redis.hset(f"entity:{entity_id}", mapping=entity.dict())
    logfire.info(f"Entity {entity_id} updated in Redis.")

    # Optionally, update entity in Supabase for persistent storage
    supabase.table("entities").update(entity.dict()).eq("id", entity_id).execute()
    logfire.info(f"Entity {entity_id} updated in Supabase.")
    
    return entity

@router.post("/entities/{entity_id}/send_message")
async def send_message(entity_id: int, message: str):
    # Get the entity's current position and details
    entity_data = await redis.hgetall(f"entity:{entity_id}")
    if not entity_data:
        logfire.error(f"Entity {entity_id} not found for sending message.")
        raise HTTPException(status_code=404, detail="Entity not found")
    
    # Save the message in Redis for the entity
    await redis.hset(f"entity:{entity_id}", "message", message)
    logfire.info(f"Message sent by Entity {entity_id}: {message}")
    
    return {"status": "Message sent successfully", "message": message}

@router.delete("/entities/{entity_id}")
async def delete_entity(entity_id: int):
    # Delete entity from Redis
    deleted = await redis.delete(f"entity:{entity_id}")
    if deleted:
        logfire.info(f"Entity {entity_id} deleted from Redis.")
    else:
        logfire.warning(f"Attempted to delete non-existent Entity {entity_id} from Redis.")
    
    # Optionally, delete entity from Supabase
    supabase.table("entities").delete().eq("id", entity_id).execute()
    logfire.info(f"Entity {entity_id} deleted from Supabase.")
    
    return {"status": "Entity deleted successfully"}

@router.get("/entities/{entity_id}/nearby", response_model=List[Entity])
async def get_nearby_entities(entity_id: int):
    # Get the entity's position from Redis
    entity_data = await redis.hgetall(f"entity:{entity_id}")
    if not entity_data:
        logfire.error(f"Entity {entity_id} not found for fetching nearby entities.")
        raise HTTPException(status_code=404, detail="Entity not found")

    try:
        entity = Entity(
            id=int(entity_data["id"]),
            name=entity_data["name"],
            x=int(entity_data["x"]),
            y=int(entity_data["y"]),
            memory=entity_data.get("memory", "")
        )
    except (KeyError, ValueError) as e:
        logfire.error(f"Error parsing entity {entity_id} data: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving entity data")

    # Fetch all entities except the current one
    all_entities = []
    for i in range(NUM_ENTITIES):
        if i != entity_id:
            entity_info = await redis.hgetall(f"entity:{i}")
            if entity_info and all(k in entity_info for k in ["id", "name", "x", "y"]):
                try:
                    nearby_entity = Entity(
                        id=int(entity_info["id"]),
                        name=entity_info["name"],
                        x=int(entity_info["x"]),
                        y=int(entity_info["y"]),
                        memory=entity_info.get("memory", "")
                    )
                    all_entities.append(nearby_entity)
                except (KeyError, ValueError) as e:
                    logfire.warning(f"Missing or invalid data for entity {i}: {e}. Skipping.")
            else:
                logfire.warning(f"Missing or incomplete data for entity {i}. Skipping.")

    # Filter nearby entities based on Chebyshev distance
    nearby_entities = [
        a for a in all_entities
        if chebyshev_distance(entity.x, entity.y, a.x, a.y) <= CHEBYSHEV_DISTANCE
    ]

    logfire.debug(f"Fetched {len(nearby_entities)} nearby entities for Entity {entity_id}.")
    return nearby_entities

@router.post("/sync_entities")
async def sync_entities():
    all_entities = []
    for i in range(NUM_ENTITIES):
        entity_data = await redis.hgetall(f"entity:{i}")
        if entity_data and all(k in entity_data for k in ["id", "name", "x", "y"]):
            try:
                entity = Entity(
                    id=int(entity_data["id"]),
                    name=entity_data["name"],
                    x=int(entity_data["x"]),
                    y=int(entity_data["y"]),
                    memory=entity_data.get("memory", "")
                )
                all_entities.append(entity)
            except (KeyError, ValueError) as e:
                logfire.warning(f"Error parsing entity {i} data: {e}. Skipping.")
        else:
            logfire.warning(f"Missing or incomplete data for entity {i}. Skipping.")

    for entity in all_entities:
        supabase.table("entities").upsert(entity.dict()).execute()
        logfire.info(f"Entity {entity.id} synchronized to Supabase.")

    logfire.info("All entities synchronized between Redis and Supabase.")
    return {"status": "Entities synchronized between Redis and Supabase"}

@router.get("/settings", response_model=SimulationSettings)
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
    logfire.debug("Simulation settings retrieved.")
    return settings

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

    logfire.info("Simulation settings updated.")
    return settings

# Endpoint to get current prompt templates
@router.get("/prompts", response_model=PromptSettings)
async def get_prompts():
    prompts = PromptSettings(
        message_generation_prompt=DEFAULT_MESSAGE_GENERATION_PROMPT,
        memory_generation_prompt=DEFAULT_MEMORY_GENERATION_PROMPT,
        movement_generation_prompt=DEFAULT_MOVEMENT_GENERATION_PROMPT
    )
    logfire.debug("Prompt templates retrieved.")
    return prompts

# Endpoint to set new prompt templates
@router.post("/prompts", response_model=PromptSettings)
async def set_prompts(prompts: PromptSettings):
    global DEFAULT_MESSAGE_GENERATION_PROMPT, DEFAULT_MEMORY_GENERATION_PROMPT, DEFAULT_MOVEMENT_GENERATION_PROMPT

    DEFAULT_MESSAGE_GENERATION_PROMPT = prompts.message_generation_prompt
    DEFAULT_MEMORY_GENERATION_PROMPT = prompts.memory_generation_prompt
    DEFAULT_MOVEMENT_GENERATION_PROMPT = prompts.movement_generation_prompt

    logfire.info("Prompt templates updated.")
    return prompts
