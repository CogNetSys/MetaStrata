import os
import io
import random
import asyncio
import json
import httpx
import logging
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time
import traceback
import tracemalloc
from pydantic_ai.models.groq import GroqModel, GroqModelName
from endpoints.database import redis, supabase
from utils import add_log, LOG_QUEUE, logger, submit_summary
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, File, UploadFile, APIRouter, Query
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from redis.asyncio import Redis
from supabase import create_client, Client
from asyncio import Semaphore
from collections import deque
from datetime import datetime
from endpoints import router
from pydantic_ai import Agent
from world import World
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
    LOG_FILE,
    LOG_QUEUE_MAX_SIZE,
    LOG_LEVEL,
    MTNN_API_ENDPOINT,
    NUM_WORLDS,
    create_world_config
)

load_dotenv()
installed_plugins = []

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

# Data Models
class StepRequest(BaseModel):
    steps: int

# Logging Configuration
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("simulation_app")

# Use the dynamic log file path
log_file_path = LOG_FILE

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

def construct_prompt(template, entity, messages):
    sanitized_messages = [msg.replace("\n", " ").replace("\"", "'").strip() for msg in messages]
    messages_str = "\n".join(sanitized_messages) if sanitized_messages else "No recent messages."
    memory = entity.get("memory", "No prior memory.").replace("\n", " ").replace("\"", "'")

    return template.format(
        entityId=entity["id"], x=entity["x"], y=entity["y"],
        grid_description=GRID_DESCRIPTION, memory=memory,
        messages=messages_str, distance=CHEBYSHEV_DISTANCE
    )

# Helper function to fetch Prompts from FastAPI
async def fetch_prompts_from_fastapi():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:8000/customization/prompts")
            if response.status_code == 200:
                return response.json()  # Return the fetched prompts
            else:
                logger.warning("Failed to fetch prompts, using default ones.")
                return {}  # Return an empty dict to trigger the default prompts
        except Exception as e:
            logger.error(f"Error fetching prompts from FastAPI: {e}")
            return {}  # Return an empty dict to trigger the default prompts

# Helper Function to add logs to the queue
def add_log(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    LOG_QUEUE.append(formatted_message)
    if len(LOG_QUEUE) > LOG_QUEUE_MAX_SIZE:
        LOG_QUEUE.popleft()  # Maintain the queue size
    logger.info(formatted_message)  # Log to the standard logger as well

# Helper function for WOrlds. Randomly selects a movement action for an agent.
def determine_agent_movement(agent: Dict) -> Dict:
    """
    Determine the movement for an agent.
    """
    # Example: Random movement
    movement_options = [
        {"x": 1, "y": 0},  # x+1
        {"x": -1, "y": 0},  # x-1
        {"x": 0, "y": 1},  # y+1
        {"x": 0, "y": -1},  # y-1
        {"x": 0, "y": 0}   # stay
    ]
    movement = random.choice(movement_options)
    return movement

# Helper function for WOrlds. Randomly generates messages between agents with a certain probability.
def generate_communications(world: World) -> List[Dict]:
    """
    Generate communications between agents.
    """
    messages = []
    for agent in world.agents:
        if random.random() < 0.1:  # 10% chance to send a message
            recipient = random.choice(world.agents)
            if recipient["id"] != agent["id"]:
                message = {
                    "sender_id": agent["id"],
                    "recipient_id": recipient["id"],
                    "content": f"Hello from Agent {agent['id']}!"
                }
                messages.append(message)
                # Log the communication
                add_log(f"Agent {agent['id']} sent message to Agent {recipient['id']}.")
    return messages


async def send_llm_request(prompt):
    global stop_signal
    async with global_request_semaphore:
        if stop_signal:
            logger.info("Stopping LLM request due to stop signal.")
            return {"message": "", "memory": "", "movement": "stay"}

        try:
            # Log the prompt for debugging
            logger.debug(f"Sending prompt to LLM:\n{prompt}")

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
        await redis.delete(f"entity:{entity['id']}:messages")  # Clear message queue

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

#World Simulation Functions

async def simulate_world_step(world: World):
    """
    Update the state of the world for one simulation step.
    This includes updating agents, tasks, resources, and handling communications.
    """
    try:
        # Example: Update agent positions
        for agent in world.agents:
            move = determine_agent_movement(agent)  # Implement this function
            agent["x"] = (agent["x"] + move["x"]) % world.grid_size
            agent["y"] = (agent["y"] + move["y"]) % world.grid_size

        # Example: Update tasks
        for task in world.tasks:
            task["progress"] += task.get("increment", 0.1)
            task["progress"] = min(task["progress"], 1.0)  # Cap at 100%

        # Example: Update resources
        consumed = world.resources.get("consumed", 0) + 1  # Increment consumption
        world.resources["consumed"] = consumed

        # Example: Handle communications
        messages = generate_communications(world)  # Implement this function
        world.communications.extend(messages)

        logger.debug(f"World {world.world_id} state updated.")
    except Exception as e:
        logger.error(f"Error simulating world {world.world_id} step: {e}")
        add_log(f"Error simulating world {world.world_id} step: {e}")

async def submit_summary(world_id: int, summary_vector: List[float]):
    """
    Send the summary vector to the mTNN for higher-level processing.
    Implement the communication with the mTNN here.
    """
    try:
        # Example: Send via HTTP POST
        async with httpx.AsyncClient() as client:
            response = await client.post(
                MTNN_API_ENDPOINT,  # Define this in your config
                json={"world_id": world_id, "summary": summary_vector}
            )
            if response.status_code == 200:
                logger.info(f"Successfully sent summary for World {world_id} to mTNN.")
            else:
                logger.warning(f"Failed to send summary for World {world_id} to mTNN. Status code: {response.status_code}")
    except Exception as e:
        logger.error(f"Error sending summary for World {world_id} to mTNN: {e}")
        add_log(f"Error sending summary for World {world_id} to mTNN: {e}")

async def fetch_tasks_for_world(world_id: int) -> List[Dict]:
    """
    Fetch or initialize tasks for the given world.
    """
    # Placeholder implementation
    tasks = []
    for i in range(100):  # Example: 100 tasks
        task = {
            "id": i,
            "progress": 0.0,
            "duration": np.random.randint(1, 10),
            "increment": 0.1  # Progress increment per step
        }
        tasks.append(task)
    logger.info(f"Fetched {len(tasks)} tasks for World {world_id}.")
    return tasks

async def fetch_resources_for_world(world_id: int) -> Dict:
    """
    Fetch or initialize resources for the given world.
    """
    # Placeholder implementation
    resources = {
        "total": 1000,
        "consumed": 0,
        "distribution": {agent_id: 100 for agent_id in range(NUM_ENTITIES)}
    }
    logger.info(f"Fetched resources for World {world_id}.")
    return resources

# Lifespan Context Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    global stop_signal
    logger.info("Starting application lifespan...")
    app.state.worlds = []
    try:
        # Startup logic
        await redis.ping()  # Asynchronous ping
        logger.info("Redis connection established.")
        # entities = await intialize_entities()

        for world_id in range(NUM_WORLDS):
            # Initialize tasks for the world
            tasks = [
                {
                    "id": i,
                    "progress": 0.0,
                    "duration": random.randint(5, 20),
                    "increment": 0.1
                }
                for i in range(100)  # Assuming 100 tasks per world
            ]

            # Initialize resources for the world
            resources = {
                "total": 1000,
                "consumed": 0,
                "distribution": {agent_id: 100 for agent_id in range(NUM_ENTITIES)}
            }

            # Create the world configuration
            world_config = create_world_config(
                world_id=world_id,
                grid_size=GRID_SIZE,
                num_agents=NUM_ENTITIES,
                tasks=tasks,  # Pass initialized tasks
                resources=resources  # Pass initialized resources
            )

            logger.debug(f"World Config: {world_config}")  # Log the world_config to inspect

            # Create and append the world object
            world = World(**world_config)  # Ensure the World class matches the configuration structure
            app.state.worlds.append(world)
            logger.info(f"World {world.world_id} initialized with {world.num_agents} agents.")
        yield

    finally:
        # Shutdown logic
        logger.info("Shutting down application...")
        stop_signal = True  # Ensure simulation stops if running
        await redis.close()  # Asynchronous close
        logger.info("Redis connection closed.")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Worlds Designer", 
    version="0.0.3", 
    description="API for World Simulations.",
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
        # swagger_css_url="https://cdn.jsdelivr.net/gh/Itz-fork/Fastapi-Swagger-UI-Dark/assets/swagger_ui_dark.min.css"
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
        </script>
    '''
    modified_html = html.body.decode("utf-8").replace("</body>", f"{custom_script}{log_area}</body>")

    return HTMLResponse(modified_html)

# Redis client initialization is handled via lifespan context manager
# Remove redundant startup and shutdown event handlers to prevent conflicts

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

            # Perform simulation logic for the world
            for world in app.state.worlds:
                await simulate_world_step(world)

                summary_vector = world.summarize_state()
                await submit_summary(world.world_id, summary_vector)

                add_log(f"World {world.world_id} summary: {summary_vector}")

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
            entities_data = await asyncio.gather(*[redis.hgetall(key) for key in valid_entity_keys])
            entities = []
            for entity_data in entities_data:
                if entity_data:
                    # Convert bytes to strings if necessary
                    if isinstance(entity_data, dict):
                        entity = {
                            "id": int(entity_data.get("id", 0)),
                            "name": entity_data.get("name", ""),
                            "x": int(entity_data.get("x", 0)),
                            "y": int(entity_data.get("y", 0)),
                            "memory": entity_data.get("memory", "")
                        }
                        entities.append(entity)
                    else:
                        add_log(f"Invalid entity data: {entity_data}")

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
                    if "message" in message_result and message_result["message"]:
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
                    if "memory" in memory_result and memory_result["memory"]:
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

        # Generate Movement (Per-Entity Processing)
        for entity in entities:
            try:
                # Construct the movement prompt for the current entity
                movement_prompt = construct_prompt(
                    prompts.get("movement_generation_prompt", DEFAULT_MOVEMENT_GENERATION_PROMPT),
                    entity,
                    []
                )

                # Send the prompt to the LLM for generating movement
                movement_result = await send_llm_request(movement_prompt)
                movement = movement_result.get("movement", "stay").strip()

                # Validate the movement response
                valid_movements = {"x+1", "x-1", "y+1", "y-1", "stay"}
                if movement not in valid_movements:
                    movement = "stay"  # Default to "stay" if the response is invalid

                # Apply the movement to the entity
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
                logger.error(f"Error during movement generation for Entity {entity['id']}: {str(e)}")
                add_log(f"Error during movement generation for Entity {entity['id']}: {str(e)}")
            
            # Optional: Throttle requests to prevent overwhelming the LLM
            await asyncio.sleep(REQUEST_DELAY)

        logger.info(f"Completed {request.steps} step(s).")
        add_log(f"Simulation steps completed: {request.steps} step(s).")
        return JSONResponse({"status": f"Performed {request.steps} step(s)."})

    except Exception as e:
        logger.error(f"Unexpected error during simulation steps: {str(e)}")
        add_log(f"Unexpected error during simulation steps: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during simulation steps: {str(e)}")

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
    DIRECTIONS: Append a memory to their existing memory field. 
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
    Get entities within the messaging range of a specific entity.
    DIRECTIONS: Enter the integer of the entity you wish to message and execute. 
    The response body reveals any entities within the messaging range of the entity you wish to message. 
    Note the "id" of the entity and then use "Send Message" to send a message as the "id" of that entity so your chosen entity receives the message. 
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
        all_entities_data = await asyncio.gather(*[
            redis.hgetall(f"entity:{i}") for i in range(NUM_ENTITIES) if i != entity_id
        ])
        all_entities = []
        for data in all_entities_data:
            if data:
                all_entities.append(Entity(**data))

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
            message_key = f"entity:{entity_id}:messages"
            await redis.lpush(message_key, message)

        # Log the broadcast action
        add_log(f"Broadcast message to all entities: {message}")
        return {"status": "success", "message": f"Broadcasted message to {len(entity_keys)} entities."}
    except Exception as e:
        error_message = f"Error broadcasting message: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.delete("/entities/{entity_id}/messages", tags=["Entity Messaging"])
async def clear_all_messages(entity_id: int):
    """
    Remove all messages for a specific entity.
    """
    try:
        message_key = f"entity:{entity_id}:messages"

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
