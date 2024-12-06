import os
import random
import asyncio
import json
import time
import tracemalloc
from typing import List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, File, UploadFile, APIRouter, Query
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from redis.asyncio import Redis
from supabase import create_client, Client
import httpx
import logging
from asyncio import Semaphore
from contextlib import asynccontextmanager
import traceback
from collections import deque
from datetime import datetime
from endpoints import router
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel, GroqModelName
import matplotlib.pyplot as plt
import numpy as np
import io
import networkx as nx
from config import (
    SUPABASE_URL,
    SUPABASE_KEY,
    GROQ_API_KEY,
    GROQ_API_ENDPOINT,
    REDIS_ENDPOINT,
    REDIS_PASSWORD,
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

load_dotenv()
installed_plugins = []

# Global log queue for real-time logging
LOG_QUEUE = deque(maxlen=200)  # Keeps the last 200 log messages

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

# Entity Model (Pydantic)
class Entity(BaseModel):
    id: int
    name: str
    x: int
    y: int
    memory: str = ""  # Default empty memory for new entities

# LLM Response Model
class LLMResponse(BaseModel):
    message: Optional[str] = ""
    memory: Optional[str] = ""
    movement: Optional[str] = "stay"

class BatchMessage(BaseModel):
    entity_id: int
    message: str

class BatchMessagesPayload(BaseModel):
    messages: List[BatchMessage]

# Redis & Supabase Initialization
redis = Redis(
    host=REDIS_ENDPOINT,
    port=6379,
    password=REDIS_PASSWORD,
    decode_responses=True,
    ssl=True  # Enable SSL/TLS
)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simulation_app")

# Semaphore for throttling concurrent requests
global_request_semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)

# Stop signal
stop_signal = False

# Define GroqModel and Agent for pydantic_ai
groq_model = GroqModel(
    model_name=LLM_MODEL,
    api_key=GROQ_API_KEY
)

agent = Agent(
    model=groq_model,
    result_type=LLMResponse,
    retries=30,
    defer_model_check=True,
)

# Fetching Nearby Entities Function
def chebyshev_distance(x1, y1, x2, y2):
    return max(abs(x1 - x2), abs(y1 - y2))

# Data Models
class StepRequest(BaseModel):
    steps: int

def construct_prompt(template, entity, messages):
    # Truncate or sanitize messages if necessary
    sanitized_messages = [msg.replace("\n", " ").strip() for msg in messages]
    messages_str = "\n".join(sanitized_messages) if sanitized_messages else "No recent messages."
    memory = entity.get("memory", "No prior memory.")

    return template.format(
        entityId=entity["id"], x=entity["x"], y=entity["y"],
        grid_description=GRID_DESCRIPTION, memory=memory,
        messages=messages_str, distance=CHEBYSHEV_DISTANCE
    )

# Helper function to fetch Prompts from FastAPI
async def fetch_prompts_from_fastapi():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/prompts")
        if response.status_code == 200:
            return response.json()  # Return the fetched prompts
        else:
            logger.warning("Failed to fetch prompts, using default ones.")
            return {}  # Return an empty dict to trigger the default prompts

# Helper Function to add logs to the queue
def add_log(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    LOG_QUEUE.append(formatted_message)
    logger.info(formatted_message)  # Log to the standard logger as well

# Modified send_llm_request using pydantic_ai's Agent
async def send_llm_request(prompt):
    global stop_signal
    async with global_request_semaphore:
        if stop_signal:
            logger.info("Stopping LLM request due to stop signal.")
            return {"message": "", "memory": "", "movement": "stay"}

        try:
            # Run the prompt through pydantic_ai's Agent
            result = await agent.run(prompt)
            response = result.data.dict()

            # Validate expected keys and provide defaults if necessary
            return {
                "message": response.get("message", ""),
                "memory": response.get("memory", ""),
                "movement": response.get("movement", "stay")
            }
        except Exception as e:
            logger.error(f"Error during LLM request: {e}")
            return {"message": "", "memory": "", "movement": "stay"}

async def intialize_entities():
    logger.info("Resetting simulation state.")
    await redis.flushdb()  # Clear all Redis data, including entity_keys

    entities = [
        {
            "id": i,
            "name": f"Entity-{i}",
            "x": random.randint(0, GRID_SIZE - 1),
            "y": random.randint(0, GRID_SIZE - 1),
            "memory": ""
        }
        for i in range(NUM_ENTITIES)
    ]

    for entity in entities:
        entity_key = f"entity:{entity['id']}"
        await redis.hset(entity_key, mapping=entity)
        await redis.lpush("entity_keys", entity_key)  # Add to entity_keys list
        await redis.delete(f"{entity['id']}:messages")  # Clear message queue

    logger.info("Entities initialized.")
    return entities

async def fetch_nearby_messages(entity, entities, message_to_send=None):
    nearby_entities = [
        a for a in entities if a["id"] != entity["id"] and chebyshev_distance(entity["x"], entity["y"], a["x"], a["y"]) <= CHEBYSHEV_DISTANCE
    ]
    received_messages = []

    for nearby_entity in nearby_entities:
        # Fetch existing messages from the nearby entity
        msg = await redis.hget(f"entity:{nearby_entity['id']}", "message")
        logger.info(f"Fetched message for entity {nearby_entity['id']}: {msg}")
        if msg:
            received_messages.append(msg)

        # If a message is being sent, add it to the recipient's queue
        if message_to_send:
            recipient_key = f"entity:{nearby_entity['id']}:messages"
            await redis.lpush(recipient_key, f"From Entity {entity['id']}: {message_to_send}")
            logger.info(f"Sent message from Entity {entity['id']} to Entity {nearby_entity['id']}")

    return received_messages

# Lifespan Context Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    global stop_signal
    logger.info("Starting application lifespan...")
    try:
        # Startup logic
        await redis.ping()
        logger.info("Redis connection established.")
        yield  # Application is running
    finally:
        # Shutdown logic
        logger.info("Shutting down application...")
        stop_signal = True  # Ensure simulation stops if running
        await redis.close()
        logger.info("Redis connection closed.")

# Create FastAPI app with lifespan
app = FastAPI(
    title="DevAGI Designer", 
    version="0.0.3", 
    description="API for world simulation and managing WebSocket logs.",
    docs_url=None, 
    lifespan=lifespan
)

# Include the router
app.include_router(router)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/docs", tags=["Simulation"], include_in_schema=False)
async def custom_swagger_ui_html():
    logger.info("Custom /docs endpoint is being served")  # Logs when /docs is accessed
    html = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="FastAPI - WebSocket Integration",
    )

    custom_script = '<script src="/static/js/websocket_client.js"></script>'
    log_area = '''
        <div class="opblock opblock-get" id="websocket-logs-section" style="margin-top: 20px; max-width: 1400px; margin: auto;">
            <div class="opblock-summary opblock-summary-get">
                <button
                    aria-expanded="true"
                    class="opblock-summary-control"
                    style="display: flex; align-items: center; justify-content: space-between; width: 100%; border: none; background: none; padding: 10px 15px; cursor: pointer; font-weight: bold; font-size: 14px;"
                    onclick="toggleLogSection()"
                >
                    <span>Simulation Live Stream</span>
                    <span style="font-size: 18px;">&#x25B2;</span>
                </button>
            </div>
            <div id="websocket-logs-container" class="opblock-body" style="display: block; padding: 10px; background-color: #f6f6f6; border: 1px solid #e8e8e8; border-radius: 4px;">
                <div id="websocket-controls" style="text-align: center; margin-bottom: 10px;">
                    <button id="reconnect-stream">Reconnect Live Stream</button>
                    <button id="select-all">Select All</button>
                    <button id="copy-logs">Copy Logs</button>
                    <button id="clear-logs">Clear Logs</button>
                </div>
                <div id="websocket-logs" style="background: #f9f9f9; padding: 10px; height: 700px; overflow-y: auto; border: 1px solid #ccc; border-radius: 5px">
                    <p>WebSocket logs will appear here...</p>
                </div>
            </div>
        </div>
    '''
    custom_script += '''
        <script>
            function toggleLogSection() {
                const container = document.getElementById('websocket-logs-container');
                const button = container.previousElementSibling.querySelector('span:nth-child(2)');
                if (container.style.display === 'none') {
                    container.style.display = 'block';
                    button.innerHTML = '&#x25B2;';
                } else {
                    container.style.display = 'none';
                    button.innerHTML = '&#x25BC;';
                }
            }

            // Automatically collapse the schemas section and open WebSocket logs on page load
            document.addEventListener("DOMContentLoaded", function() {
                // Collapse the schemas section
                const schemasSummary = document.querySelector("section.models.is-open .models-summary");
                if (schemasSummary) {
                    schemasSummary.click(); // Simulates a click to collapse it
                }

                // Open the WebSocket logs section
                const logsContainer = document.getElementById("websocket-logs-container");
                const logsButton = logsContainer.previousElementSibling.querySelector('span:nth-child(2)');
                if (logsContainer && logsButton) {
                    logsContainer.style.display = 'block';
                    logsButton.innerHTML = '&#x25B2;'; // Ensure the arrow points up
                }

                // Add event listener for the clear logs button
                const clearLogsButton = document.getElementById("clear-logs");
                const logsDiv = document.getElementById("websocket-logs");

                if (clearLogsButton) {
                    clearLogsButton.addEventListener("click", function() {
                        if (logsDiv) {
                            logsDiv.innerHTML = ""; // Clear the logs
                            logsDiv.innerHTML = "<p>WebSocket logs will appear here...</p>"; // Reset placeholder
                        }
                    });
                }

                // Add event listener for the select all button
                const selectAllButton = document.getElementById("select-all");
                if (selectAllButton) {
                    selectAllButton.addEventListener("click", function() {
                        const range = document.createRange();
                        range.selectNodeContents(logsDiv);
                        const selection = window.getSelection();
                        selection.removeAllRanges();
                        selection.addRange(range);
                    });
                }

                document.addEventListener("DOMContentLoaded", function () {
                // Get all buttons inside the controls section
                const buttons = document.querySelectorAll("#websocket-controls button");

                buttons.forEach(button => {
                    // Add mouseover effect
                    button.addEventListener("mouseover", function () {
                        button.style.backgroundColor = "#ddd"; // Light gray background on hover
                        button.style.cursor = "pointer"; // Pointer cursor on hover
                    });

                    // Add mouseout effect (reset to original color)
                    button.addEventListener("mouseout", function () {
                        if (button.id === "clear-logs") {
                            button.style.backgroundColor = "#f44336"; // Red for Clear Logs
                        } else if (button.id === "select-all") {
                            button.style.backgroundColor = "#4caf50"; // Green for Select All
                        } else if (button.id === "copy-logs") {
                            button.style.backgroundColor = "#2196f3"; // Blue for Copy Logs
                        } else if (button.id === "reconnect-stream") {
                            button.style.backgroundColor = "#9e9e9e"; // Gray for Reconnect
                        }
                    });

                    // Add click effect
                    button.addEventListener("click", function () {
                        button.style.backgroundColor = "#aaa"; // Darker shade on click
                        setTimeout(function () {
                            // Reset background after click
                            if (button.id === "clear-logs") {
                                button.style.backgroundColor = "#f44336";
                            } else if (button.id === "select-all") {
                                button.style.backgroundColor = "#4caf50";
                            } else if (button.id === "copy-logs") {
                                button.style.backgroundColor = "#2196f3";
                            } else if (button.id === "reconnect-stream") {
                                button.style.backgroundColor = "#9e9e9e";
                            }
                        }, 200); // Reset after 200ms
                    });
                });
            });

            });
        </script>
    '''
    modified_html = html.body.decode("utf-8").replace("</body>", f"{custom_script}{log_area}</body>")
    
    return HTMLResponse(modified_html)

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": f"Custom Error: {exc.detail}"}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    # Log the exception details for debugging
    logger.error(f"Unhandled exception: {str(exc)}\n{traceback.format_exc()}")

    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred. Please try again later."}
    )

@app.post("/step", tags=["World Simulation"])
async def perform_steps(request: StepRequest):
    global stop_signal
    stop_signal = False  # Reset stop signal before starting steps

    try:
        add_log(f"Simulation steps requested: {request.steps} step(s).")
        
        # Fetch the current prompt templates from FastAPI
        prompts = await fetch_prompts_from_fastapi()
        add_log("Fetched prompt templates successfully.")

        logger.info("Starting simulation steps...")

        for step in range(request.steps):
            if stop_signal:
                add_log("Simulation steps halted by stop signal.")
                break

            # Fetch all entity keys dynamically from Redis
            entity_keys = await redis.keys("entity:*")  # Match all entity keys
            if not entity_keys:
                add_log("No entities found in Redis. Aborting simulation steps.")
                return JSONResponse({"status": "No entities to process."})

            logger.info(f"Step {step + 1}: Found {len(entity_keys)} entities.")
            add_log(f"Step {step + 1}: Found {len(entity_keys)} entities.")

            # Filter keys to ensure only valid hashes are processed
            valid_entity_keys = []
            for key in entity_keys:
                key_type = await redis.type(key)
                if key_type == "hash":
                    valid_entity_keys.append(key)
                else:
                    add_log(f"Skipping invalid key {key} of type {key_type}")

            # Fetch entity data from Redis for all valid keys
            entities = [
                {
                    "id": int(entity_data["id"]),
                    "name": entity_data["name"],
                    "x": int(entity_data["x"]),
                    "y": int(entity_data["y"]),
                    "memory": entity_data.get("memory", "")
                }
                for entity_data in await asyncio.gather(*[redis.hgetall(key) for key in valid_entity_keys])
                if entity_data  # Ensure we only include valid entity data
            ]

            logger.info(f"Processing {len(entities)} entities.")
            add_log(f"Processing {len(entities)} entities for step {step + 1}.")

            # Process incoming messages for each entity
            for entity in entities:
                try:
                    # Fetch the existing message field
                    message = await redis.hget(f"entity:{entity['id']}", "message")

                    if message:
                        logger.info(f"Entity {entity['id']} received message: {message}")
                        add_log(f"Entity {entity['id']} received message: {message}")

                        # Optionally update memory or trigger actions based on the message
                        updated_memory = f"{entity['memory']} | Received: {message}"
                        await redis.hset(f"entity:{entity['id']}", "memory", updated_memory)

                        # Clear the message field after processing (if required)
                        await redis.hset(f"entity:{entity['id']}", "message", "")
                except Exception as e:
                    logger.error(f"Error processing message for Entity {entity['id']}: {str(e)}")
                    add_log(f"Error processing message for Entity {entity['id']}: {str(e)}")

            # Clear message queues only after processing all entities
            for entity in entities:
                await redis.delete(f"entity:{entity['id']}:messages")

            # Message Generation
            for entity in entities:
                try:
                    messages = await fetch_nearby_messages(entity, entities)
                    message_prompt = prompts.get("message_generation_prompt", DEFAULT_MESSAGE_GENERATION_PROMPT)
                    message_result = await send_llm_request(
                        construct_prompt(message_prompt, entity, messages)
                    )
                    if "message" in message_result:
                        await redis.hset(f"entity:{entity['id']}", "message", message_result["message"])
                        add_log(f"Message generated for Entity {entity['id']}: {message_result['message']}")
                except Exception as e:
                    logger.error(f"Error generating message for Entity {entity['id']}: {str(e)}")
                    add_log(f"Error generating message for Entity {entity['id']}: {str(e)}")
                await asyncio.sleep(REQUEST_DELAY)

            if stop_signal:
                logger.info("Stopping after message generation due to stop signal.")
                add_log("Simulation steps halted after message generation by stop signal.")
                break

            # Memory Generation
            for entity in entities:
                try:
                    messages = await fetch_nearby_messages(entity, entities)
                    memory_prompt = prompts.get("memory_generation_prompt", DEFAULT_MEMORY_GENERATION_PROMPT)
                    memory_result = await send_llm_request(
                        construct_prompt(memory_prompt, entity, messages)
                    )
                    if "memory" in memory_result:
                        await redis.hset(f"entity:{entity['id']}", "memory", memory_result["memory"])
                        add_log(f"Memory updated for Entity {entity['id']}: {memory_result['memory']}")
                except Exception as e:
                    logger.error(f"Error generating memory for Entity {entity['id']}: {str(e)}")
                    add_log(f"Error generating memory for Entity {entity['id']}: {str(e)}")
                await asyncio.sleep(REQUEST_DELAY)

            if stop_signal:
                logger.info("Stopping after memory generation due to stop signal.")
                add_log("Simulation steps halted after memory generation by stop signal.")
                break

            # Movement Generation
            for entity in entities:
                try:
                    movement_prompt = prompts.get("movement_generation_prompt", DEFAULT_MOVEMENT_GENERATION_PROMPT)
                    movement_result = await send_llm_request(
                        construct_prompt(movement_prompt, entity, [])
                    )
                    if "movement" in movement_result:
                        movement = movement_result["movement"].strip().lower()
                        initial_position = (entity["x"], entity["y"])

                        # Apply movement logic
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
                            add_log(f"Entity {entity['id']} stays in place at {initial_position}.")
                            continue
                        else:
                            logger.warning(f"Invalid movement command for Entity {entity['id']}: {movement}")
                            add_log(f"Invalid movement command for Entity {entity['id']}: {movement}")
                            continue

                        # Log and update position
                        logger.info(f"Entity {entity['id']} moved from {initial_position} to ({entity['x']}, {entity['y']}) with action '{movement}'.")
                        add_log(f"Entity {entity['id']} moved from {initial_position} to ({entity['x']}, {entity['y']}) with action '{movement}'.")
                        await redis.hset(f"entity:{entity['id']}", mapping={"x": entity["x"], "y": entity["y"]})
                except Exception as e:
                    logger.error(f"Error generating movement for Entity {entity['id']}: {str(e)}")
                    add_log(f"Error generating movement for Entity {entity['id']}: {str(e)}")
                await asyncio.sleep(REQUEST_DELAY)

        logger.info(f"Completed {request.steps} step(s).")
        add_log(f"Simulation steps completed: {request.steps} step(s).")
        return JSONResponse({"status": f"Performed {request.steps} step(s)."})

    except Exception as e:
        logger.error(f"Unexpected error during simulation steps: {str(e)}")
        add_log(f"Unexpected error during simulation steps: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during simulation steps: {str(e)}")
    
@app.get("/entities", tags=["Entities"])
async def list_all_entities():
    """
    Fetch details of all entities, including their current position, memory, and messages.
    """
    try:
        # Fetch all entity keys from Redis
        entity_keys = await redis.keys("entity:*")
        if not entity_keys:
            add_log("No entities found in Redis.")
            return []

        # Fetch and parse entity data, including messages
        entities = []
        for key in entity_keys:
            entity_data = await redis.hgetall(key)
            if not entity_data:
                continue

            # Fetch the current message for the entity
            entity_id = int(entity_data["id"])
            message = entity_data.get("message", "")

            # Append the entity with messages to the result
            entities.append({
                "id": entity_id,
                "name": entity_data.get("name"),
                "x": int(entity_data.get("x")),
                "y": int(entity_data.get("y")),
                "memory": entity_data.get("memory", ""),
                "messages": [message] if message else [],  # Wrap in a list for consistency
            })

        add_log(f"Fetched details for {len(entities)} entities, including messages.")
        return entities
    except Exception as e:
        error_message = f"Error fetching all entities: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.post("/entities", response_model=Entity, tags=["Entities"])
async def create_entity(entity: Entity):
    entity_key = f"entity:{entity.id}"

    try:
        # Log the attempt to create a new entity
        add_log(f"Attempting to create new entity with ID {entity.id} and name '{entity.name}'.")

        # Check if the ID already exists in Redis
        if await redis.exists(entity_key):
            error_message = f"Entity ID {entity.id} already exists in Redis."
            add_log(error_message)
            raise HTTPException(status_code=400, detail=error_message)

        # Check if the ID already exists in Supabase
        existing_entity = supabase.table("entities").select("id").eq("id", entity.id).execute()
        if existing_entity.data:
            error_message = f"Entity ID {entity.id} already exists in Supabase."
            add_log(error_message)
            raise HTTPException(status_code=400, detail=error_message)

        # Save entity data in Redis
        await redis.hset(entity_key, mapping=entity.dict())
        add_log(f"Entity data for ID {entity.id} saved in Redis.")

        # Add the entity key to the Redis list
        await redis.lpush("entity_keys", entity_key)
        add_log(f"Entity key for ID {entity.id} added to Redis entity_keys list.")

        # Save the entity in Supabase
        supabase.table("entities").insert(entity.dict()).execute()
        add_log(f"Entity data for ID {entity.id} saved in Supabase.")

        # Log the successful creation of the entity
        add_log(f"New entity created successfully: ID={entity.id}, Name={entity.name}, Position=({entity.x}, {entity.y})")
        
        return entity

    except Exception as e:
        # Log and raise unexpected errors
        error_message = f"Error creating entity with ID {entity.id}: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/entities/{entity_id}", response_model=Entity, tags=["Entities"])
async def get_entity(entity_id: int):
    # Log the attempt to fetch entity data
    add_log(f"Fetching data for entity with ID {entity_id}.")

    # Fetch entity data from Redis
    entity_data = await redis.hgetall(f"entity:{entity_id}")
    if not entity_data:
        error_message = f"Entity with ID {entity_id} not found."
        add_log(error_message)
        raise HTTPException(status_code=404, detail=error_message)

    # Convert Redis data into an Entity model (ensure proper types are cast)
    try:
        entity = Entity(
            id=entity_id,
            name=entity_data.get("name"),
            x=int(entity_data.get("x")),
            y=int(entity_data.get("y")),
            memory=entity_data.get("memory", "")
        )
        add_log(f"Successfully fetched entity with ID {entity_id}, Name: {entity.name}, Position: ({entity.x}, {entity.y}).")
        return entity
    except Exception as e:
        error_message = f"Failed to parse data for entity ID {entity_id}: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.put("/entities/{entity_id}", response_model=Entity, tags=["Entities"])
async def update_entity(entity_id: int, entity: Entity):
    try:
        # Log the attempt to update entity data
        add_log(f"Attempting to update entity with ID {entity_id}.")

        # Update entity data in Redis
        await redis.hset(f"entity:{entity_id}", mapping=entity.dict())
        add_log(f"Entity with ID {entity_id} updated in Redis.")

        # Update entity in Supabase
        supabase.table("entities").update(entity.dict()).eq("id", entity_id).execute()
        add_log(f"Entity with ID {entity_id} updated in Supabase.")

        # Log successful update
        add_log(f"Successfully updated entity with ID {entity_id}.")
        return entity
    except Exception as e:
        # Log and raise error if update fails
        error_message = f"Error updating entity with ID {entity_id}: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.delete("/entities/{entity_id}", tags=["Entities"])
async def delete_entity(entity_id: int):
    entity_key = f"entity:{entity_id}"
    try:
        # Log the attempt to delete an entity
        add_log(f"Attempting to delete entity with ID {entity_id}.")

        # Delete entity from Redis
        await redis.delete(entity_key)
        add_log(f"Entity with ID {entity_id} deleted from Redis.")

        # Remove the key from the Redis list
        await redis.lrem("entity_keys", 0, entity_key)
        add_log(f"Entity key with ID {entity_id} removed from Redis entity_keys list.")

        # Optionally, delete the entity from Supabase
        supabase.table("entities").delete().eq("id", entity_id).execute()
        add_log(f"Entity with ID {entity_id} deleted from Supabase.")

        return {"status": "Entity deleted successfully"}
    except Exception as e:
        # Log and raise any unexpected errors
        error_message = f"Error deleting entity with ID {entity_id}: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/entities/{entity_id}/messages", response_model=Optional[str], tags=["Entity Messaging"])
async def retrieve_message(entity_id: int):
    """
    Retrieve the current message for a specific entity.
    """
    try:
        # Validate that the entity exists
        entity_key = f"entity:{entity_id}"
        if not await redis.exists(entity_key):
            error_message = f"Entity with ID {entity_id} not found."
            add_log(error_message)
            raise HTTPException(status_code=404, detail=error_message)

        # Fetch the message field for the entity
        message = await redis.hget(entity_key, "message")
        if not message:
            add_log(f"No current message found for Entity {entity_id}.")
            return None  # Return None if no message exists

        add_log(f"Retrieved current message for Entity {entity_id}: \"{message}\".")
        return message
    except Exception as e:
        error_message = f"Error retrieving message for Entity {entity_id}: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)
   
@app.post("/entities/{recipient_id}/create_memory", tags=["Entity Messaging"])
async def create_memory(
    recipient_id: int,
    message: str = Query(..., description="The memory content to add or update for the recipient entity.")
):
    """
    Create a memory for an entity. 
    DIRECTIONS: Append a memory to thier existing memory field. 
    Enter the integer of the Entity you wish to add a memory to and the memory content to add for the recipient entity.
    """
    recipient_key = f"entity:{recipient_id}"

    try:
        # Log the attempt to create memory
        add_log(f"Creating a memory for Entity {recipient_id}: \"{message}\".")

        # Validate that the recipient exists
        if not await redis.exists(recipient_key):
            error_message = f"Recipient Entity ID {recipient_id} not found."
            add_log(error_message)
            raise HTTPException(status_code=404, detail=error_message)

        # Fetch the recipient's existing memory field
        existing_memory = await redis.hget(recipient_key, "memory")
        if existing_memory:
            # Append the new message to the recipient's memory
            updated_memory = f"{existing_memory}\n{message}"
        else:
            # Start the memory with the new message
            updated_memory = message

        # Update the recipient's memory field
        await redis.hset(recipient_key, "memory", updated_memory)
        add_log(f"Memory updated successfully for Entity {recipient_id}: \"{message}\".")

        return {"status": "Memory updated successfully", "message": message}

    except Exception as e:
        # Log and raise any unexpected errors
        error_message = f"Error creating memory for Entity {recipient_id}: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)
    
@app.post("/entities/{recipient_id}/send_message", tags=["Entity Messaging"])
async def send_message(
    recipient_id: int,
    message: str = Query(..., description="The message to send to the recipient entity.")
):
    """
    Send a message as an entity.
    DIRECTIONS: Append a message to a designated entity's existing message field.
    This means you are sending a message as the designated entity to the surrounding entities. 
    The designated entity does not receive the message.
    """
    recipient_key = f"entity:{recipient_id}"

    try:
        # Log the attempt to send a message
        add_log(f"Attempting to send a message to Entity {recipient_id}: \"{message}\".")

        # Validate that the recipient exists
        if not await redis.exists(recipient_key):
            error_message = f"Recipient Entity ID {recipient_id} not found."
            add_log(error_message)
            add_log(f"Entity {recipient_id} not found.")
            raise HTTPException(status_code=404, detail=error_message)

        # Fetch the existing message (if any)
        existing_message = await redis.hget(recipient_key, "message") or ""

        # Append the new message, separating with a delimiter if needed
        updated_message = existing_message + "\n" + message if existing_message else message

        # Update the `message` field with the appended message
        await redis.hset(recipient_key, "message", updated_message)

        # Log the successful message update
        add_log(f"Message appended successfully to Entity {recipient_id}: \"{message}\".")

        return {"status": "Message appended successfully", "recipient_id": recipient_id, "message": updated_message}

    except Exception as e:
        # Log and raise any unexpected errors
        error_message = f"Error sending message to Entity {recipient_id}: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/entities/{entity_id}/nearby", response_model=List[Entity], tags=["Entity Messaging"])
async def get_nearby_entities(entity_id: int):
    """
    Send a message to an entity. 
    DIRECTIONS: Enter the integer of the entity you wish to message and execute. 
    The response body reveals any entities within the messaging range of the entity you wish to message. 
    Note the "id" of the entity and then use "Send Messsage" to send a message as the "id" of that entity so your chosen entity recieves the message. 
    The sending entity will not have a memory of the message being sent.
    """
    try:
        # Log the attempt to fetch nearby entities
        add_log(f"Fetching nearby entities for Entity ID {entity_id}.")

        # Get the entity's position from Redis
        entity_data = await redis.hgetall(f"entity:{entity_id}")
        if not entity_data:
            error_message = f"Entity with ID {entity_id} not found."
            add_log(error_message)
            raise HTTPException(status_code=404, detail=error_message)

        entity = Entity(**entity_data)

        # Fetch all entities except the current one
        all_entities = [
            Entity(**await redis.hgetall(f"entity:{i}"))
            for i in range(NUM_ENTITIES) if i != entity_id
        ]

        # Filter nearby entities based on Chebyshev distance
        nearby_entities = [
            a for a in all_entities
            if chebyshev_distance(entity.x, entity.y, a.x, a.y) <= CHEBYSHEV_DISTANCE
        ]

        add_log(f"Nearby entities fetched successfully for Entity ID {entity_id}. Total nearby entities: {len(nearby_entities)}.")
        return nearby_entities
    except Exception as e:
        # Log and raise any unexpected errors
        error_message = f"Error fetching nearby entities for Entity ID {entity_id}: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.post("/entities/update_batch_memory", tags=["Entity Messaging"])
async def update_batch_memory(payload: BatchMessagesPayload):
    """
    Update the `memory` field of multiple entities in one request.

    This function appends the provided content to the `memory` field of the respective entities in Redis.
    """
    try:
        updated_entities = []
        for msg in payload.messages:
            entity_key = f"entity:{msg.entity_id}"
            
            # Validate that the entity exists
            if not await redis.exists(entity_key):
                add_log(f"Entity {msg.entity_id} not found. Skipping update.")
                continue
            
            # Fetch the current memory for the entity (if it exists)
            existing_memory = await redis.hget(entity_key, "memory") or ""
            
            # Append the new content to the existing memory, separated by a newline
            updated_memory = f"{existing_memory}\n{msg.message}".strip()
            
            # Update the `memory` field with the appended content
            await redis.hset(entity_key, "memory", updated_memory)
            updated_entities.append(msg.entity_id)
        
        add_log(f"Batch memory updates applied to Entities: {updated_entities}.")
        return {"status": "success", "updated_entities": updated_entities}
    
    except Exception as e:
        error_message = f"Custom Error: Error updating batch memory: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.delete("/entities/{entity_id}/memory", tags=["Entity Messaging"])
async def clear_memory(entity_id: int):
    """
    Wipe an entity's memory field.
    """
    try:
        entity_key = f"entity:{entity_id}"
        
        # Validate entity exists
        if not await redis.exists(entity_key):
            error_message = f"Entity with ID {entity_id} not found."
            add_log(error_message)
            raise HTTPException(status_code=404, detail=error_message)
        
        # Clear the memory field
        await redis.hset(entity_key, "memory", "")
        add_log(f"Memory cleared for Entity {entity_id}.")
        return {"status": "success", "message": f"Memory cleared for Entity {entity_id}."}
    except Exception as e:
        error_message = f"Error clearing memory for Entity {entity_id}: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.post("/entities/broadcast_message", tags=["Entity Messaging"])
async def broadcast_message(message: str):
    """
    Broadcast a message to all entities.
    """
    try:
        # Fetch all entity keys
        entity_keys = await redis.keys("entity:*")
        if not entity_keys:
            add_log("No entities found to broadcast the message.")
            raise HTTPException(status_code=404, detail="No entities found.")

        # Broadcast the message to all entities
        for key in entity_keys:
            entity_id = key.split(":")[1]  # Extract entity ID
            message_key = f"{entity_id}:messages"
            await redis.lpush(message_key, message)

        # Log the broadcast action
        add_log(f"Broadcast message to all entities: {message}")
        return {"status": "success", "message": f"Broadcasted message to {len(entity_keys)} entities."}
    except Exception as e:
        error_message = f"Error broadcasting message: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

class BatchMessage(BaseModel):
    entity_id: int
    message: str

class BatchMessagesPayload(BaseModel):
    messages: List[BatchMessage]

@app.delete("/entities/{entity_id}/messages", tags=["Entity Messaging"])
async def clear_all_messages(entity_id: int):
    """
    Remove all messages for a specific entity.
    """
    try:
        message_key = f"{entity_id}:messages"

        # Validate that the message key exists
        if not await redis.exists(message_key):
            error_message = f"No messages found for Entity {entity_id}."
            add_log(error_message)
            raise HTTPException(status_code=404, detail=error_message)

        # Clear all messages
        await redis.delete(message_key)
        add_log(f"All messages cleared for Entity {entity_id}.")
        return {"status": "success", "message": f"All messages cleared for Entity {entity_id}."}
    except Exception as e:
        error_message = f"Error clearing messages for Entity {entity_id}: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/analytics", tags=["Utilities"])
async def analytics_dashboard():
    """
    Retrieve aggregated statistics about the simulation.
    """
    try:
        entity_keys = await redis.keys("entity:*")
        total_entities = len(entity_keys)
        total_messages = 0
        memory_sizes = []

        for key in entity_keys:
            entity_data = await redis.hgetall(key)
            messages = await redis.lrange(f"{entity_data['id']}:messages", 0, -1)
            total_messages += len(messages)
            memory_sizes.append(len(entity_data.get("memory", "")))

        avg_memory_size = sum(memory_sizes) / len(memory_sizes) if memory_sizes else 0
        avg_messages = total_messages / total_entities if total_entities > 0 else 0

        analytics = {
            "total_entities": total_entities,
            "total_messages": total_messages,
            "average_memory_size": avg_memory_size,
            "average_messages_per_entity": avg_messages,
        }

        add_log("Analytics dashboard data generated successfully.")
        return analytics
    except Exception as e:
        error_message = f"Error generating analytics dashboard: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/simulation/state/download", tags=["Utilities"])
async def download_simulation_state():
    """
    Download the current state of the simulation as a JSON file.
    """
    try:
        # Collect all entities from Redis
        entity_keys = await redis.keys("entity:*")
        entities = []
        for key in entity_keys:
            entity_data = await redis.hgetall(key)
            messages = await redis.lrange(f"{entity_data['id']}:messages", 0, -1)
            entity_data["messages"] = messages
            entities.append(entity_data)

        # Write to a temporary JSON file
        file_path = "/tmp/simulation_state.json"
        with open(file_path, "w") as f:
            json.dump(entities, f)

        add_log("Simulation state saved to JSON file.")
        return FileResponse(file_path, media_type="application/json", filename="simulation_state.json")
    except Exception as e:
        error_message = f"Error downloading simulation state: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.post("/simulation/state/upload", tags=["Utilities"])
async def upload_simulation_state(state_file: UploadFile = File(...)):
    """
    Upload and restore a saved simulation state.
    """
    try:
        # Read uploaded file
        state_data = json.loads(await state_file.read())

        # Clear current Redis data
        await redis.flushdb()
        add_log("Redis database cleared before state restoration.")

        # Restore entities and messages
        for entity in state_data:
            entity_key = f"entity:{entity['id']}"
            messages_key = f"{entity['id']}:messages"
            messages = entity.pop("messages", [])
            await redis.hset(entity_key, mapping=entity)
            for message in messages:
                await redis.lpush(messages_key, message)

        add_log("Simulation state restored successfully.")
        return {"status": "Simulation state uploaded and restored."}
    except Exception as e:
        error_message = f"Error uploading simulation state: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.post("/sync_entity", tags=["Utilities"])
async def sync_entity():
    try:
        # Log the start of the synchronization process
        add_log("Synchronization process initiated between Redis and Supabase.")

        all_entities = [
            Entity(**await redis.hgetall(f"entity:{i}"))
            for i in range(NUM_ENTITIES)
        ]

        for entity in all_entities:
            supabase.table("entities").upsert(entity.dict()).execute()
            add_log(f"Entity with ID {entity.id} synchronized to Supabase.")

        add_log("Synchronization process completed successfully.")
        return {"status": "Entities synchronized between Redis and Supabase"}
    except Exception as e:
        # Log and raise any unexpected errors
        error_message = f"Error during entity synchronization: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)
    
@app.get("/debug/redis", tags=["Debugging and Optimization"])
async def test_redis_connectivity():
    """
    Test and report the status of Redis connectivity.
    """
    try:
        # Ping Redis to check connectivity
        ping_response = await redis.ping()
        if ping_response:
            add_log("Redis connectivity test successful.")
            return {"status": "connected", "message": "Redis is reachable."}
        else:
            raise Exception("Redis ping returned a falsy value.")
    except Exception as e:
        error_message = f"Redis connectivity test failed: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/debug/supabase", tags=["Debugging and Optimization"])
async def test_supabase_connectivity():
    """
    Test and report the status of Supabase connectivity.
    """
    try:
        # Test a basic query to check Supabase connectivity
        response = supabase.table("entities").select("*").limit(1).execute()
        if response.data:
            add_log("Supabase connectivity test successful.")
            return {"status": "connected", "message": "Supabase is reachable."}
        else:
            raise Exception("No data returned, but connection is established.")
    except Exception as e:
        error_message = f"Supabase connectivity test failed: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

app.get("/debug/performance", tags=["Debugging and Optimization"])
async def profile_simulation_performance():
    """
    Profile the simulation to identify bottlenecks in processing or memory usage.
    """
    try:
        tracemalloc.start()
        start_time = time.time()

        # Example: Simulate fetching all entities (you can replace this with actual simulation logic)
        entity_keys = await redis.keys("entity:*")
        for key in entity_keys:
            await redis.hgetall(key)

        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        performance_data = {
            "execution_time": f"{end_time - start_time:.2f} seconds",
            "memory_usage": f"{current / 10**6:.2f} MB",
            "peak_memory_usage": f"{peak / 10**6:.2f} MB"
        }

        add_log(f"Performance profiling completed: {performance_data}")
        return JSONResponse(performance_data)
    except Exception as e:
        error_message = f"Error during performance profiling: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/debug/redis/{key}", tags=["Debugging and Optimization"])
async def inspect_redis_key(key: str):
    """
    Fetch the content of a specific Redis key for debugging purposes. Format: "entity:0"
    """
    try:
        key_type = await redis.type(key)
        if key_type == "none":
            error_message = f"Redis key '{key}' does not exist."
            add_log(error_message)
            raise HTTPException(status_code=404, detail=error_message)

        if key_type == "string":
            value = await redis.get(key)
        elif key_type == "list":
            value = await redis.lrange(key, 0, -1)
        elif key_type == "hash":
            value = await redis.hgetall(key)
        else:
            value = f"Unsupported key type: {key_type}"

        add_log(f"Inspected Redis key '{key}': {value}")
        return {"key": key, "type": key_type, "value": value}
    except Exception as e:
        error_message = f"Error inspecting Redis key '{key}': {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/audit", tags=["Debugging and Optimization"])
async def fetch_audit_trail():
    """
    Fetch a detailed log of all significant actions performed during the simulation.
    """
    try:
        if not LOG_QUEUE:
            add_log("Audit trail requested, but no logs are available.")
            return {"audit_trail": []}

        return {"audit_trail": list(LOG_QUEUE)}
    except Exception as e:
        error_message = f"Error fetching audit trail: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/grid", tags=["Visualization"])
async def generate_grid_visualization():
    """
    Generate a grid visualization displaying entities' numbers within their respective cells.
    """
    try:
        # Fetch all entities from Redis
        entity_keys = await redis.keys("entity:*")
        if not entity_keys:
            raise HTTPException(status_code=404, detail="No entities found.")

        # Get grid size from configuration
        grid_size = GRID_SIZE  # Assume GRID_SIZE is defined in your settings

        # Create a blank grid
        grid = np.full((grid_size, grid_size), "", dtype=object)

        # Place entities on the grid
        for key in entity_keys:
            entity_data = await redis.hgetall(key)
            x, y, entity_id = int(entity_data["x"]), int(entity_data["y"]), entity_data["id"]
            grid[y][x] = str(entity_id)  # Place the entity number at the position

        # Plot the grid
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks(np.arange(0, grid_size, 1))
        ax.set_yticks(np.arange(0, grid_size, 1))
        ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
        ax.tick_params(which="major", bottom=False, left=False, labelbottom=False, labelleft=False)

        # Annotate with entity numbers
        for y in range(grid_size):
            for x in range(grid_size):
                if grid[y][x]:
                    ax.text(x + 0.5, grid_size - y - 0.5, grid[y][x], ha="center", va="center", color="blue")

        # Save the grid to a BytesIO stream
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        # Return the image as a StreamingResponse
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        error_message = f"Error generating grid visualization: {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)
    
@app.get("/visualization/heatmap", tags=["Visualization"])
async def agent_location_heatmap():
    """
    Generate a heatmap of agent locations.
    """
    try:
        # Fetch all entities from Redis
        entity_keys = await redis.keys("entity:*")
        if not entity_keys:
            raise HTTPException(status_code=404, detail="No entities found.")

        # Initialize the grid
        heatmap = np.zeros((GRID_SIZE, GRID_SIZE))

        # Increment the grid based on agent locations
        for key in entity_keys:
            entity_data = await redis.hgetall(key)
            x, y = int(entity_data["x"]), int(entity_data["y"])
            heatmap[y][x] += 1

        # Create a heatmap visualization
        fig, ax = plt.subplots(figsize=(8, 8))
        cax = ax.matshow(heatmap, cmap="viridis", origin="lower")
        plt.colorbar(cax)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        error_message = f"Error generating heatmap: {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/visualization/trajectory", tags=["Visualization"])
async def trajectory_visualization():
    try:
        # Retrieve entity keys
        entity_keys = await redis.keys("entity:*")
        if not entity_keys:
            return {"error": "No entities found for trajectory visualization."}

        # Fetch entity positions
        trajectories = []
        for key in entity_keys:
            entity_data = await redis.hgetall(key)
            if entity_data:
                trajectories.append((int(entity_data["x"]), int(entity_data["y"])))

        # Generate plot
        plt.figure(figsize=(10, 10))
        x_vals, y_vals = zip(*trajectories) if trajectories else ([], [])
        plt.plot(x_vals, y_vals, marker='o')
        plt.title("Agent Trajectories")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)

        # Return the plot as an image
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        return {"message": f"Custom Error: Error generating trajectory visualization: {str(e)}"}

@app.get("/visualization/network", tags=["Visualization"])
async def interaction_network_graph():
    """
    Generate a graph visualization of agent interactions.
    """
    try:
        # Fetch interactions from Redis
        interactions = []  # Replace with logic to fetch agent interactions
        graph = nx.Graph()

        for interaction in interactions:
            entity_a, entity_b = interaction["from"], interaction["to"]
            graph.add_edge(entity_a, entity_b)

        # Draw the graph
        fig, ax = plt.subplots(figsize=(10, 10))
        nx.draw_networkx(graph, ax=ax, node_size=700, font_size=10)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        error_message = f"Error generating interaction graph: {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/visualization/behavior", tags=["Visualization"])
async def behavior_over_time():
    """
    Generate a time-series visualization of agent behavior.
    """
    try:
        # Fetch behavior data from Redis
        behavior_data = {}  # Replace with logic to fetch behavior data

        # Prepare the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for agent_id, data in behavior_data.items():
            times, values = zip(*data)
            ax.plot(times, values, label=f"Agent {agent_id}")

        ax.set_title("Behavior Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Behavior Metric")
        ax.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        error_message = f"Error generating behavior plot: {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/visualization/task_allocation", tags=["Visualization"])
async def task_allocation_chart():
    try:
        # Retrieve task allocation data
        task_data = await redis.hgetall("tasks")  # Replace with appropriate Redis call
        if not task_data:
            return {"message": "No task allocation data found."}

        # Verify task_data is a dictionary
        if isinstance(task_data, list):
            raise ValueError("Task data is a list; expected a dictionary.")

        # Prepare data for visualization
        tasks = list(task_data.keys())
        allocations = [len(eval(entities)) for entities in task_data.values()]  # Example: Eval if data is stored as strings

        # Generate bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(tasks, allocations, color="skyblue")
        plt.title("Task Allocation Chart")
        plt.xlabel("Tasks")
        plt.ylabel("Number of Assigned Entities")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Return the chart as an image
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        return {"message": f"Custom Error: Error generating task allocation chart: {str(e)}"}

@app.get("/visualization/system_metrics", tags=["Visualization"])
async def system_metrics_over_time():
    """
    Generate a time-series visualization of system-wide metrics.
    """
    try:
        # Fetch system metrics data
        metrics = {}  # Replace with logic to fetch system metrics

        # Prepare the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for metric_name, data in metrics.items():
            times, values = zip(*data)
            ax.plot(times, values, label=metric_name)

        ax.set_title("System Metrics Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Metric Value")
        ax.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        error_message = f"Error generating system metrics plot: {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)

# A list to store custom rules
custom_rules = []

from fastapi.responses import HTMLResponse

@app.get("/custom_rules_docs", tags=["Docs"])
async def custom_rules_docs():
    """
    Comprehensive inline documentation for custom rules creation and usage.
    """
    html_content = """
    <html>
        <head>
            <title>Custom Rules Documentation</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 20px;
                    background-color: #f9f9f9;
                    color: #333;
                }
                h1 {
                    color: #2c3e50;
                }
                h2 {
                    margin-top: 20px;
                    color: #34495e;
                }
                pre {
                    background: #f4f4f4;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    overflow-x: auto;
                }
                .example, .note {
                    padding: 15px;
                    border-radius: 4px;
                }
                .example {
                    background-color: #eaf7ff;
                    border-left: 4px solid #3498db;
                }
                .note {
                    background-color: #fff8e1;
                    border-left: 4px solid #f39c12;
                }
                a {
                    color: #2980b9;
                    text-decoration: none;
                }
                a:hover {
                    text-decoration: underline;
                }
            </style>
        </head>
        <body>
            <h1>Custom Rules Documentation</h1>
            <p>Define specific behaviors for your simulation entities using custom rules. These rules allow dynamic adjustments to entity and environment behavior.</p>

            <h2>Getting Started</h2>
            <p>Write rules in Python and use the following objects for logic:</p>
            <ul>
                <li><strong>entity:</strong> Represents the current agent in the simulation.</li>
                <li><strong>simulation:</strong> Represents the overall simulation environment.</li>
                <li><strong>log(message):</strong> Log messages to the simulation's audit trail.</li>
            </ul>

            <h2>Examples</h2>
            <div class="example">
                <h3>1. Boundary Check</h3>
                <p>Add a memory entry when an entity reaches the boundary:</p>
                <pre>
if entity.x == 0 or entity.x == simulation.width - 1:
    entity.memory.append("Boundary reached")
                </pre>
            </div>
            <div class="example">
                <h3>2. Collision Detection</h3>
                <p>Log interactions when two entities collide:</p>
                <pre>
if entity1.x == entity2.x and entity1.y == entity2.y:
    log(f"Entity {entity1.id} collided with Entity {entity2.id}")
                </pre>
            </div>

            <h2>System Variables</h2>
            <ul>
                <li><strong>entity:</strong> Access the entity's attributes such as <code>entity.x</code>, <code>entity.y</code>, and <code>entity.memory</code>.</li>
                <li><strong>simulation:</strong> Use properties like <code>simulation.width</code> and <code>simulation.height</code> to reference the environment.</li>
            </ul>

            <h2>Validation</h2>
            <div class="note">
                <strong>Important:</strong> All rules are validated for syntax and restricted to safe operations. Avoid using:
                <ul>
                    <li>File system operations (e.g., open, write).</li>
                    <li>External network calls.</li>
                    <li>Execution of arbitrary system commands.</li>
                </ul>
            </div>

            <h2>How to Use</h2>
            <ol>
                <li>Go to the <a href="/docs">API Documentation</a>.</li>
                <li>Use the <code>/simulation/custom_rules</code> endpoint to add rules.</li>
                <li>Submit a JSON payload like this:
                <pre>
{
    "rule": "if entity.x == 10: entity.memory.append('At boundary')"
}
                </pre>
                </li>
            </ol>

            <h2>Management</h2>
            <ul>
                <li><strong>Add a Rule:</strong> Use <code>POST /simulation/custom_rules</code>.</li>
                <li><strong>List All Rules:</strong> Use <code>GET /simulation/custom_rules</code>.</li>
                <li><strong>Delete a Rule:</strong> Use <code>DELETE /simulation/custom_rules/{rule_id}</code>.</li>
            </ul>

            <h2>Advanced Use Cases</h2>
            <ul>
                <li>Trigger events based on conditions like proximity or thresholds.</li>
                <li>Implement multi-agent collaboration through shared states or memory.</li>
                <li>Use rules to modify environment behavior dynamically.</li>
            </ul>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/simulation/custom_rules", tags=["Customization"])
async def add_custom_rule(rule: str):
    """
    Add or update custom rules for the simulation.

    <br>
    <h2>Custom Rules Documentation</h2>
    <p>
        Custom rules allow you to extend the simulation behavior dynamically.
        Use Python syntax and predefined objects such as <code>entity</code> and <code>environment</code>.
    </p>
    <h3>Quick Instructions</h3>
    <ul>
        <li>Write rules in valid Python syntax.</li>
        <li>Use the <code>entity</code> object to access or modify individual agent properties.</li>
        <li>Use the <code>environment</code> object to interact with the simulation's environment.</li>
    </ul>
    <h3>Example Rule</h3>
    <pre>
    if entity.x == 10:
        entity.memory += 'At boundary'
    </pre>
    <p>
        For more details, visit the full 
        <a href="http://127.0.0.1:8000/custom_rules_docs" target="_blank">Custom Rules Documentation</a>.
    </p>
    """
    try:
        # Validate the rule (basic check for malicious code)
        if not rule or not isinstance(rule, str):
            raise HTTPException(status_code=400, detail="Invalid rule format.")

        # Example: Simple validation for Python syntax
        try:
            compile(rule, "<string>", "exec")
        except SyntaxError as e:
            raise HTTPException(status_code=400, detail=f"Invalid Python syntax: {str(e)}")

        # Add the rule to the custom rules list
        custom_rules.append(rule)

        return {"status": "Rule added successfully", "rule": rule, "total_rules": len(custom_rules)}

    except Exception as e:
        return {"message": f"Error adding custom rule: {str(e)}"}

from fastapi.responses import HTMLResponse

installed_plugins = []

@app.get("/plugins-docs", tags=["Docs"])
async def plugins_docs():
    """
    Comprehensive inline documentation for plugin management.
    """
    html_content = """
    <html>
        <head>
            <title>Plugin Management Documentation</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 20px;
                    background-color: #f4f4f9;
                    color: #333;
                }
                h1 {
                    color: #0056b3;
                }
                h2 {
                    margin-top: 20px;
                    color: #0066cc;
                }
                pre {
                    background: #eee;
                    padding: 10px;
                    border: 1px solid #ccc;
                    overflow-x: auto;
                }
                .example {
                    background-color: #e8f6fc;
                    padding: 15px;
                    border-left: 4px solid #00aaff;
                }
                .note {
                    background-color: #fff4e5;
                    padding: 15px;
                    border-left: 4px solid #ffa726;
                }
            </style>
        </head>
        <body>
            <h1>Plugin Management Documentation</h1>
            <p>Manage plugins or extensions for your simulation environment. Plugins allow you to extend the functionality of your simulation by adding custom features or behaviors.</p>

            <h2>Supported Actions</h2>
            <p>The following actions are supported for plugin management:</p>
            <ul>
                <li><strong>Install:</strong> Adds a plugin to the simulation.</li>
                <li><strong>Uninstall:</strong> Removes a plugin from the simulation.</li>
                <li><strong>List:</strong> Lists all currently installed plugins.</li>
            </ul>

            <h2>Example Payloads</h2>
            <div class="example">
                <h3>1. Installing a Plugin</h3>
                <pre>
{
    "plugin_name": "custom_logger",
    "action": "install"
}
                </pre>
                <p>This example installs a plugin named <code>custom_logger</code>.</p>
            </div>
            <div class="example">
                <h3>2. Uninstalling a Plugin</h3>
                <pre>
{
    "plugin_name": "custom_logger",
    "action": "uninstall"
}
                </pre>
                <p>This example removes the plugin named <code>custom_logger</code>.</p>
            </div>
            <div class="example">
                <h3>3. Listing Installed Plugins</h3>
                <pre>
{
    "action": "list"
}
                </pre>
                <p>This example retrieves a list of all currently installed plugins.</p>
            </div>

            <h2>Usage Instructions</h2>
            <ol>
                <li>Go to the <a href="/docs">API Documentation</a>.</li>
                <li>Use the <code>/simulation/plugins</code> endpoint.</li>
                <li>Submit a JSON payload with the desired action and plugin name (if applicable).</li>
            </ol>

            <h2>Validation Rules</h2>
            <p>The following validation rules are applied for plugin management:</p>
            <ul>
                <li><strong>Action:</strong> Must be one of <code>install</code>, <code>uninstall</code>, or <code>list</code>.</li>
                <li><strong>Plugin Name:</strong> Required for <code>install</code> and <code>uninstall</code> actions.</li>
                <li><strong>Unique Plugins:</strong> Plugins must have unique names. Duplicate installations are not allowed.</li>
            </ul>

            <div class="note">
                <strong>Note:</strong> Plugin functionality depends on the specific implementation of each plugin. Ensure that any custom plugins are compatible with the simulation environment.
            </div>

            <h2>Advanced Use Cases</h2>
            <ul>
                <li>Create a logging plugin to track simulation events in real-time.</li>
                <li>Develop an AI plugin to enable agents to make intelligent decisions.</li>
                <li>Integrate visualization plugins for enhanced simulation analysis.</li>
            </ul>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/simulation/plugins", tags=["Customization"])
async def manage_plugins(plugin_name: str = None, action: str = None):
    """
    Manage plugins or extensions for the simulation environment.

    <br>
    <h2>Plugin Management Documentation</h2>
    <p>
        Plugins extend the simulation's capabilities. This endpoint allows you to install, uninstall, or list plugins.
    </p>
    <h3>Quick Instructions</h3>
    <ul>
        <li>Use the "install" action to add a plugin.</li>
        <li>Use the "uninstall" action to remove a plugin.</li>
        <li>Use the "list" action to see all installed plugins.</li>
    </ul>
    <h3>Example Payloads</h3>
    <ul>
        <li><b>Install:</b> <code>{"plugin_name": "custom_logger", "action": "install"}</code></li>
        <li><b>Uninstall:</b> <code>{"plugin_name": "custom_logger", "action": "uninstall"}</code></li>
        <li><b>List:</b> <code>{"action": "list"}</code></li>
    </ul>
    <p>
        For more details, visit the full 
        <a href="http://127.0.0.1:8000/plugins-docs" target="_blank">Plugin Management Documentation</a>.
    </p>
    """
    try:
        if not action or action not in ["install", "uninstall", "list"]:
            raise HTTPException(status_code=400, detail="Invalid action. Supported actions: install, uninstall, list.")

        if action == "list":
            return {"installed_plugins": installed_plugins}

        if action in ["install", "uninstall"] and not plugin_name:
            raise HTTPException(status_code=400, detail="Plugin name is required for install or uninstall actions.")

        if action == "install":
            if plugin_name in installed_plugins:
                raise HTTPException(status_code=400, detail=f"Plugin '{plugin_name}' is already installed.")
            installed_plugins.append(plugin_name)
            return {"status": "Plugin installed successfully", "plugin_name": plugin_name}

        if action == "uninstall":
            if plugin_name not in installed_plugins:
                raise HTTPException(status_code=400, detail=f"Plugin '{plugin_name}' is not installed.")
            installed_plugins.remove(plugin_name)
            return {"status": "Plugin uninstalled successfully", "plugin_name": plugin_name}

    except Exception as e:
        return {"message": f"Error managing plugins: {str(e)}"}

# WebSocket Endpoint for Real-Time Logs
@app.websocket("/logs")
async def websocket_log_endpoint(websocket: WebSocket):
    add_log("WebSocket connection initiated for real-time logs.")
    await websocket.accept()
    try:
        while True:
            if LOG_QUEUE:
                # Send all current logs to the client
                for log in list(LOG_QUEUE):
                    await websocket.send_text(log)
                LOG_QUEUE.clear()  # Clear the queue after sending
            await asyncio.sleep(1)  # Check for new logs every second
    except asyncio.CancelledError:
        # Log graceful cancellation
        add_log("WebSocket log stream cancelled by the client.")
    except Exception as e:
        # Log unexpected exceptions
        error_message = f"Error in WebSocket log stream: {str(e)}"
        logger.error(error_message)
        add_log(error_message)
    finally:
        # Log connection close and clean up
        add_log("WebSocket connection for real-time logs closed.")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)