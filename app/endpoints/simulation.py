# /app/endpoints/simulation.py

import asyncio
import httpx
import logging
from fastapi import APIRouter, WebSocket
from fastapi.responses import JSONResponse
from typing import List
from jinja2 import Environment, FileSystemLoader
import json

import logfire
from app.config import Settings, settings, calculate_chebyshev_distance
from app.utils.database import redis, supabase
from app.utils.models import Entity, StepRequest, initialize_entities, fetch_prompts_from_fastapi, fetch_nearby_messages

# ---- Import the response models ----
from app.utils.simulation_utils import MessageResponse, MemoryResponse, MovementResponse, LLMRequestError

router = APIRouter()

# Initialize Jinja2 environment
env = Environment(loader=FileSystemLoader("templates"))  # Load templates from the 'templates' directory

chebyshev_distance = settings.SIMULATION.CHEBYSHEV_DISTANCE

# Connection Pooling client for HTTPX requests
client = httpx.AsyncClient(timeout=10.0)

# Semaphore for throttling concurrent requests
global_request_semaphore = asyncio.Semaphore(settings.SIMULATION.MAX_CONCURRENT_REQUESTS)

# -------------------------------------------------------------
# HELPER FUNCTIONS SECTION
# -------------------------------------------------------------

def get_simulation_settings():
    return settings.SIMULATION

# -------------------------------------------------------------
# LOGGING SECTION
# -------------------------------------------------------------

# Configure Python's logging
logging.basicConfig(level=settings.SIMULATION.LOG_LEVEL)

# Save the original logfire.debug method
original_debug = logfire.debug

def patched_debug(message, *args, **kwargs):
    """
    Redirect logfire.debug calls to Python's logging.debug.
    """
    if settings.LOGFIRE.LOGFIRE_ENABLED:
        if args:
            message = message.format(*args, **kwargs)
        logging.debug(message)  # Send to Python logging instead of Logfire

# Patch logfire.debug
logfire.debug = patched_debug

# -------------------------------------------------------------
# FUNCTIONS SECTION
# -------------------------------------------------------------

async def send_llm_request(prompt, entity_id=None, max_retries=3, base_delay=2) -> MessageResponse | MemoryResponse | MovementResponse | LLMRequestError:
    async with global_request_semaphore:
        headers = {
            'Authorization': f'Bearer {settings.GROQ.GROQ_API_KEY.get_secret_value()}',
            'Content-Type': 'application/json'
        }

        for attempt in range(max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        f"{settings.GROQ.GROQ_API_ENDPOINT}",
                        headers=headers,
                        json={
                            "model": settings.SIMULATION.LLM_MODEL,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": settings.SIMULATION.LLM_MAX_TOKENS,
                            "temperature": settings.SIMULATION.LLM_TEMPERATURE,
                        }
                    )
                
                response.raise_for_status()  # Raise an exception for HTTP errors

                # Log the raw response for debugging
                if settings.LOGFIRE.LOGFIRE_ENABLED:
                    logfire.debug(f"Raw response from LLM (entity {entity_id}, attempt {attempt + 1}): {response.text}")

                # Add Response Validation
                response_json = response.json()
                choices = response_json.get('choices', [])
                if not choices:
                    error_msg = "No choices found in the API response."
                    if settings.LOGFIRE.LOGFIRE_ENABLED:
                        logfire.error(error_msg)
                    return LLMRequestError(error=error_msg)

                message_content = choices[0].get('message', {}).get('content', "")
                if not message_content:
                    error_msg = "No message content found in the API response."
                    if settings.LOGFIRE.LOGFIRE_ENABLED:
                        logfire.error(error_msg)
                    return LLMRequestError(error=error_msg)
                
                # Attempt to parse the message content as JSON
                try:
                    message_json = json.loads(message_content)
                except json.JSONDecodeError:
                    if settings.LOGFIRE.LOGFIRE_ENABLED:
                        logfire.error(f"Failed to parse message content as JSON: {message_content}")
                    return LLMRequestError(error="Invalid JSON format in LLM response.")

                # Determine the response type based on the presence of specific keys
                if "message" in message_json:
                    validated_response = MessageResponse(message=message_json["message"])
                elif "memory" in message_json:
                    validated_response = MemoryResponse(memory=message_json["memory"])
                elif "movement" in message_json:
                    validated_response = MovementResponse(movement=message_json["movement"])
                else:
                    if settings.LOGFIRE.LOGFIRE_ENABLED:
                        logfire.error(f"Unexpected response format from LLM: {message_json}")
                    return LLMRequestError(error="Unexpected response format from LLM.")

                await asyncio.sleep(settings.SIMULATION.REQUEST_DELAY)
                return validated_response

            except httpx.HTTPStatusError as e:
                if settings.LOGFIRE.LOGFIRE_ENABLED:
                    logfire.error(f"HTTP error during LLM request: {e}, Response: {e.response.text}")
                status = e.response.status_code
                if status in [429, 503]:
                    if attempt >= max_retries:
                        if settings.LOGFIRE.LOGFIRE_ENABLED:
                            logfire.error("Exceeded max retries for LLM request.")
                        return LLMRequestError(error="Exceeded max retries due to rate limiting or service unavailability.")
                    
                    # Use 'Retry-After' header if available
                    retry_after = e.response.headers.get('Retry-After')
                    if retry_after:
                        delay = int(retry_after)
                    else:
                        delay = base_delay * (2 ** attempt)
                    
                    if settings.LOGFIRE.LOGFIRE_ENABLED:
                        logfire.error(f"Received {status} error. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    return LLMRequestError(error=f"HTTP error during LLM request: {str(e)}")

            except Exception as e:
                if settings.LOGFIRE.LOGFIRE_ENABLED:
                    logfire.error(f"Error during LLM request: {e}")
                return LLMRequestError(error=f"Error during LLM request: {str(e)}")

        return LLMRequestError(error="Failed to get a valid response from LLM.")
    
# -------------------------------------------------------------
# SIMULATION ENDPOINTS SECTION
# -------------------------------------------------------------

@router.post("/reset_and_initialize", tags=["Simulation"])
async def reset_and_initialize():
    global stop_signal
    stop_signal = False  # Reset stop signal before starting
    await redis.flushdb()
    entities = await initialize_entities()
    if settings.LOGFIRE.LOGFIRE_ENABLED:
        logfire.info("Simulation reset and initialized successfully.")
    return JSONResponse({"status": "Simulation reset and initialized successfully.", "entities": entities})

@router.post("/step", tags=["Simulation"])
async def perform_steps(request: StepRequest):
    global stop_signal
    stop_signal = False  # Reset stop signal before starting steps

    if settings.LOGFIRE.LOGFIRE_ENABLED:
        logfire.info(f"Starting {request.steps} step(s).")

    # Fetch the current prompt templates from FastAPI
    prompts = await fetch_prompts_from_fastapi()
    if settings.LOGFIRE.LOGFIRE_ENABLED:
        logfire.debug(f"Fetched prompts: {prompts}")

    entities = []
    for i in range(settings.SIMULATION.NUM_ENTITIES):
        entity_data = await redis.hgetall(f"entity:{i}")
        if entity_data:
            entities.append({
                "id": int(entity_data["id"]),
                "name": entity_data["name"],
                "x": int(entity_data["x"]),
                "y": int(entity_data["y"]),
                "memory": entity_data.get("memory", "")
            })

    if settings.LOGFIRE.LOGFIRE_ENABLED:
        logfire.info(f"Loaded {len(entities)} entities for simulation.")

    for step in range(request.steps):
        if stop_signal:
            if settings.LOGFIRE.LOGFIRE_ENABLED:
                logfire.info("Stopping steps due to stop signal.")
            break

        # Use the appropriate template and render it with entity data
        message_template = settings.SIMULATION.DEFAULT_MESSAGE_GENERATION_PROMPT
        memory_template = settings.SIMULATION.DEFAULT_MEMORY_GENERATION_PROMPT
        movement_template = settings.SIMULATION.DEFAULT_MOVEMENT_GENERATION_PROMPT

        # --- Message Generation ---
        # Collect nearby messages for all entities
        nearby_messages_dict = {}
        for entity in entities:
            nearby_messages = await fetch_nearby_messages(entity, entities)
            nearby_messages_dict[entity["id"]] = nearby_messages

        # Collect messages for all entities concurrently
        message_tasks = []
        for entity in entities:
            messages_for_entity = nearby_messages_dict.get(entity["id"], [])
            
            # Render the message prompt
            message_prompt_formatted = message_template.render(
                entityId=entity["id"],
                x=entity["x"],
                y=entity["y"],
                grid_description=settings.SIMULATION.grid_description,
                memory=entity["memory"],
                messages="\n".join(messages_for_entity),
                distance=settings.SIMULATION.CHEBYSHEV_DISTANCE
            )

            # Create a task for message generation
            task = send_llm_request(message_prompt_formatted, entity_id=entity["id"])
            message_tasks.append(task)

        # Run all message generation tasks concurrently
        message_responses = await asyncio.gather(*message_tasks)

        # Collect messages from the responses
        messages = {}
        for i, response in enumerate(message_responses):
            if isinstance(response, MessageResponse):
                entity_id = entities[i]["id"]
                messages[entity_id] = response.message
                await redis.hset(f"entity:{entity_id}", mapping={"message": response.message})
                if settings.LOGFIRE.LOGFIRE_ENABLED:
                    logfire.info(f"Entity {entity_id} message updated to: {response.message.replace('{', '{{').replace('}', '}}')}")
            elif isinstance(response, LLMRequestError):
                if settings.LOGFIRE.LOGFIRE_ENABLED:
                    logfire.error(f"Error generating message for entity {entities[i]['id']}: {response.error}")

        # --- End of Message Generation ---

        # Update memory for all entities concurrently
        memory_tasks = [
            send_llm_request(
                memory_template.render(
                    entityId=entity["id"],
                    x=entity["x"],
                    y=entity["y"],
                    grid_description=settings.SIMULATION.grid_description,
                    memory=entity["memory"],
                    messages="\n".join(messages.get(entity["id"], [])),  # Pass all collected messages here
                    distance=settings.SIMULATION.CHEBYSHEV_DISTANCE
                ),
                entity_id=entity["id"]
            )
            for entity in entities
        ]
        memory_responses = await asyncio.gather(*memory_tasks)

        for i, response in enumerate(memory_responses):
            if isinstance(response, MemoryResponse):
                await redis.hset(f"entity:{entities[i]['id']}", "memory", response.memory)
                if settings.LOGFIRE.LOGFIRE_ENABLED:
                    logfire.info(f"Entity {entities[i]['id']} memory updated to: {response.memory.replace('{', '{{').replace('}', '}}')}")
            elif isinstance(response, LLMRequestError):
                if settings.LOGFIRE.LOGFIRE_ENABLED:
                    logfire.error(f"Error generating memory for entity {entities[i]['id']}: {response.error}")

        # Determine movement for all entities concurrently
        movement_tasks = [
            send_llm_request(
                movement_template.render(
                    entityId=entity["id"],
                    x=entity["x"],
                    y=entity["y"],
                    grid_description=settings.SIMULATION.grid_description,
                    memory=entity["memory"],
                    messages="\n".join(messages.get(entity["id"], [])),  # Pass all collected messages here
                    distance=settings.SIMULATION.CHEBYSHEV_DISTANCE
                ),
                entity_id=entity["id"]
            )
            for entity in entities
        ]
        movement_responses = await asyncio.gather(*movement_tasks)

        movements = {}
        for i, response in enumerate(movement_responses):
            if isinstance(response, MovementResponse):
                movements[entities[i]["id"]] = response.movement.strip().lower()
            elif isinstance(response, LLMRequestError):
                if settings.LOGFIRE.LOGFIRE_ENABLED:
                    logfire.error(f"Error generating movement for entity {entities[i]['id']}: {response.error}")

        # Execute movements for all entities
        for entity in entities:
            movement = movements.get(entity["id"])
            new_x, new_y = entity["x"], entity["y"]
            if movement in ["x+1", "x-1", "y+1", "y-1", "stay"]:
                if movement == "x+1":
                    new_x = (entity["x"] + 1) % settings.SIMULATION.GRID_SIZE
                elif movement == "x-1":
                    new_x = (entity["x"] - 1) % settings.SIMULATION.GRID_SIZE
                elif movement == "y+1":
                    new_y = (entity["y"] + 1) % settings.SIMULATION.GRID_SIZE
                elif movement == "y-1":
                    new_y = (entity["y"] - 1) % settings.SIMULATION.GRID_SIZE

                # Log and update position if changed
                if (new_x, new_y) != (entity["x"], entity["y"]):
                    log_message = f"Entity {entity['id']} moved from ({entity['x']}, {entity['y']}) to ({new_x}, {new_y}) with action '{movement}'."
                    if settings.LOGFIRE.LOGFIRE_ENABLED:
                        logfire.info(log_message)
                    await redis.hset(f"entity:{entity['id']}", mapping={"x": new_x, "y": new_y})
                    # Update entity's position in the entities list
                    entity["x"], entity["y"] = new_x, new_y
                elif movement == "stay":
                    if settings.LOGFIRE.LOGFIRE_ENABLED:
                        logfire.info(f"Entity {entity['id']} stays in place at ({entity['x']}, {entity['y']}).")

            else:
                if settings.LOGFIRE.LOGFIRE_ENABLED:
                    logfire.error(f"Invalid or missing movement command for Entity {entity['id']}.")

        # Generate Report - Use Instructor and Jinja2 templace to create a report.
        if settings.LOGFIRE.LOGFIRE_ENABLED:
            logfire.info(f"Generating report for step {step + 1}")

        entities_data = []
        for i in range(settings.SIMULATION.NUM_ENTITIES):
            entity_data = await redis.hgetall(f"entity:{i}")
            if entity_data:
                entities_data.append({
                    "id": int(entity_data["id"]),
                    "name": entity_data["name"],
                    "x": int(entity_data["x"]),
                    "y": int(entity_data["y"]),
                    "memory": entity_data.get("memory", ""),
                    "message": entity_data.get("message", "")
                })

        template = env.get_template("simulation_report.html")
        html_report = template.render(step_number=step + 1, entities=entities_data)

        report_filename = f"simulation_report_step_{step + 1}.html"
        with open(report_filename, "w") as f:
            f.write(html_report)

        if settings.LOGFIRE.LOGFIRE_ENABLED:
            logfire.info(f"Report saved to {report_filename}")

    if settings.LOGFIRE.LOGFIRE_ENABLED:
        logfire.info(f"Completed {request.steps} step(s).")
    return JSONResponse({"status": f"Performed {request.steps} step(s).", "report": report_filename})

@router.post("/stop", tags=["Simulation"])
async def stop_simulation():
    global stop_signal
    stop_signal = True
    if settings.LOGFIRE.LOGFIRE_ENABLED:
        logfire.info("Stop signal triggered.")
    return JSONResponse({"status": "Simulation stopping."})