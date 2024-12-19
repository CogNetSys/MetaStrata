import asyncio
import logging
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from typing import List
from fastapi.responses import JSONResponse
import httpx
import logfire
from pydantic import BaseModel
from app.models import SimulationSettings, PromptSettings
from app.main import (
    StepRequest, chebyshev_distance, redis, logger, supabase, initialize_entities,
    fetch_prompts_from_fastapi, fetch_nearby_messages, connected_clients, GROQ_API_KEY, GROQ_API_ENDPOINT,
    GRID_SIZE, NUM_ENTITIES, MAX_STEPS, CHEBYSHEV_DISTANCE, LLM_MODEL,
    LLM_MAX_TOKENS, LLM_TEMPERATURE, REQUEST_DELAY, MAX_CONCURRENT_REQUESTS,
    GRID_DESCRIPTION, DEFAULT_MESSAGE_GENERATION_PROMPT,
    DEFAULT_MEMORY_GENERATION_PROMPT, DEFAULT_MOVEMENT_GENERATION_PROMPT,
    send_log_message, LOGFIRE_ENABLED
)

# Global variable to track connected WebSocket clients
connected_clients: List[WebSocket] = []

router = APIRouter()

# Semaphore for throttling concurrent requests
global_request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


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
    if LOGFIRE_ENABLED:
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
            if LOGFIRE_ENABLED:
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
                            if LOGFIRE_ENABLED:
                                logfire.error("Exceeded max retries for LLM request.")
                            return {"message": "", "memory": "", "movement": "stay"}
                        delay = base_delay * (2 ** (attempt - 1))
                        if LOGFIRE_ENABLED:
                            logfire.error(f"Received 429 Too Many Requests. Retrying in {delay} seconds...")
                        await asyncio.sleep(delay)
                        continue
                    response.raise_for_status()
                    result = response.json()

                    # Validate expected keys
                    if not all(key in result for key in ["choices"]):
                        if LOGFIRE_ENABLED:
                            logfire.error(f"Incomplete response from LLM: {result}")
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
                            if LOGFIRE_ENABLED:
                                logfire.error(f"Invalid movement command in LLM response: {content}")
                            movement = "stay"  # Default to "stay" if invalid
                        await asyncio.sleep(REQUEST_DELAY)
                        return {"movement": movement}
                    else:
                        if LOGFIRE_ENABLED:
                            logfire.error(f"Unexpected prompt type: {prompt}")
                        await asyncio.sleep(REQUEST_DELAY)
                        return {"message": "", "memory": "", "movement": "stay"}
            except Exception as e:
                if LOGFIRE_ENABLED:
                    logfire.error(f"Error during LLM request: {e}")
                return {"message": "", "memory": "", "movement": "stay"}

# Simulation API Endpoints

@router.post("/reset_and_initialize", tags=["Simulation"])
async def reset_and_initialize():
    global stop_signal
    stop_signal = False  # Reset stop signal before starting
    await redis.flushdb()
    entities = await initialize_entities()
    if LOGFIRE_ENABLED:
        logfire.info("Simulation reset and initialized successfully.")
    return JSONResponse({"status": "Simulation reset and initialized successfully.", "entities": entities})

@router.post("/step", tags=["Simulation"])
async def perform_steps(request: StepRequest):
    global stop_signal
    stop_signal = False  # Reset stop signal before starting steps

    if LOGFIRE_ENABLED:
        logfire.info(f"Starting {request.steps} step(s).")

    # Fetch the current prompt templates from FastAPI
    prompts = await fetch_prompts_from_fastapi()
    if LOGFIRE_ENABLED:
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

    if LOGFIRE_ENABLED:
        logfire.info(f"Loaded {len(entities)} entities for simulation.")

    for _ in range(request.steps):
        if stop_signal:
            if LOGFIRE_ENABLED:
                logfire.info("Stopping steps due to stop signal.")
            break

        # Use either custom prompts from FastAPI or fall back to default prompts
        message_prompt = prompts.get("message_generation_prompt", DEFAULT_MESSAGE_GENERATION_PROMPT)
        memory_prompt = prompts.get("memory_generation_prompt", DEFAULT_MEMORY_GENERATION_PROMPT)
        movement_prompt = prompts.get("movement_generation_prompt", DEFAULT_MOVEMENT_GENERATION_PROMPT)

        # Message Generation
        for entity in entities:
            if stop_signal:
                if LOGFIRE_ENABLED:
                    logfire.info("Stopping after message generation due to stop signal.")
                break
            nearby_messages = await fetch_nearby_messages(entity, entities)
            if LOGFIRE_ENABLED:
                logfire.debug(f"Entity {entity['id']} nearby messages: {nearby_messages}")

            message_result = await send_llm_request(
                message_prompt.format(
                    entityId=entity["id"], x=entity["x"], y=entity["y"],
                    grid_description=GRID_DESCRIPTION,
                    memory=entity["memory"], messages="\n".join(nearby_messages),
                    distance=CHEBYSHEV_DISTANCE
                )
            )
            if LOGFIRE_ENABLED:
                logfire.debug(f"LLM message result for Entity {entity['id']}: {message_result}")

            if "message" not in message_result:
                if LOGFIRE_ENABLED:
                    logfire.error(f"Skipping message update for Entity {entity['id']} due to invalid response.")
                continue
            await redis.hset(f"entity:{entity['id']}", "message", message_result.get("message", ""))
            if LOGFIRE_ENABLED:
                logfire.info(f"Entity {entity['id']} message updated to: {message_result.get('message')}")
            await asyncio.sleep(REQUEST_DELAY)

        if stop_signal:
            if LOGFIRE_ENABLED:
                logfire.info("Stopping after message generation due to stop signal.")
            break

        # Memory Generation
        for entity in entities:
            if stop_signal:
                if LOGFIRE_ENABLED:
                    logfire.info("Stopping after memory generation due to stop signal.")
                break
            nearby_messages = await fetch_nearby_messages(entity, entities)
            if LOGFIRE_ENABLED:
                logfire.debug(f"Entity {entity['id']} nearby messages for memory: {nearby_messages}")

            memory_result = await send_llm_request(
                memory_prompt.format(
                    entityId=entity["id"], x=entity["x"], y=entity["y"],
                    grid_description=GRID_DESCRIPTION,
                    memory=entity["memory"], messages="\n".join(nearby_messages),
                    distance=CHEBYSHEV_DISTANCE
                )
            )
            if LOGFIRE_ENABLED:
                logfire.debug(f"LLM memory result for Entity {entity['id']}: {memory_result}")

            if "memory" not in memory_result:
                if LOGFIRE_ENABLED:
                    logfire.error(f"Skipping memory update for Entity {entity['id']} due to invalid response.")
                continue
            updated_memory = memory_result.get("memory", entity["memory"])
            await redis.hset(f"entity:{entity['id']}", "memory", updated_memory)
            if LOGFIRE_ENABLED:
                logfire.info(f"Entity {entity['id']} memory updated to: {updated_memory}")
            await asyncio.sleep(REQUEST_DELAY)

        if stop_signal:
            if LOGFIRE_ENABLED:
                logfire.info("Stopping after memory generation due to stop signal.")
            break

        # Movement Generation
        for entity in entities:
            if stop_signal:
                if LOGFIRE_ENABLED:
                    logfire.info("Stopping after movement generation due to stop signal.")
                break
            nearby_messages = await fetch_nearby_messages(entity, entities)
            if LOGFIRE_ENABLED:
                logfire.debug(f"Entity {entity['id']} nearby messages for movement: {nearby_messages}")

            movement_result = await send_llm_request(
                movement_prompt.format(
                    entityId=entity["id"], x=entity["x"], y=entity["y"],
                    grid_description=GRID_DESCRIPTION,
                    memory=entity["memory"], messages="\n".join(nearby_messages),
                    distance=CHEBYSHEV_DISTANCE
                )
            )
            if LOGFIRE_ENABLED:
                logfire.debug(f"LLM movement result for Entity {entity['id']}: {movement_result}")

            if "movement" not in movement_result:
                if LOGFIRE_ENABLED:
                    logfire.error(f"Skipping movement update for Entity {entity['id']} due to invalid response.")
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
                if LOGFIRE_ENABLED:
                    logfire.info(f"Entity {entity['id']} stays in place at {initial_position}.")
                continue  # No update needed for stay
            else:
                if LOGFIRE_ENABLED:
                    logfire.error(f"Invalid movement command for Entity {entity['id']}: {movement}")
                continue  # Skip invalid commands

            # Log and update position
            if LOGFIRE_ENABLED:
                logfire.info(f"Entity {entity['id']} moved from {initial_position} to ({entity['x']}, {entity['y']}) with action '{movement}'.")
            await redis.hset(f"entity:{entity['id']}", mapping={"x": entity["x"], "y": entity["y"]})
            if LOGFIRE_ENABLED:
                logfire.debug(f"Entity {entity['id']} position updated in Redis.")
            await asyncio.sleep(REQUEST_DELAY)

    if LOGFIRE_ENABLED:
        logfire.info(f"Completed {request.steps} step(s).")
    return JSONResponse({"status": f"Performed {request.steps} step(s)."})

@router.post("/stop", tags=["Simulation"])
async def stop_simulation():
    global stop_signal
    stop_signal = True
    if LOGFIRE_ENABLED:
        logfire.info("Stop signal triggered.")
    return JSONResponse({"status": "Simulation stopping."})
