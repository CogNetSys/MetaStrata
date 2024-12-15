import os
import random
import asyncio
from typing import List, Optional, Union
from fastapi import FastAPI, HTTPException, WebSocket, Query, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel, Field, BaseSettings, validator
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

# PydanticAI Imports
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel, GroqModelName

# -------------------------------
# Configuration Management with Pydantic Settings
# -------------------------------

class AppConfig(BaseSettings):
    supabase_url: str = Field(..., env='SUPABASE_URL')
    supabase_key: str = Field(..., env='SUPABASE_KEY')
    groq_api_key: str = Field(..., env='GROQ_API_KEY')
    groq_api_endpoint: str = Field(..., env='GROQ_API_ENDPOINT')
    redis_endpoint: str = Field(..., env='REDIS_ENDPOINT')
    redis_password: str = Field(..., env='REDIS_PASSWORD')
    grid_size: int = Field(10, env='GRID_SIZE')
    num_entities: int = Field(50, env='NUM_ENTITIES')
    max_steps: int = Field(100, env='MAX_STEPS')
    chebyshev_distance: int = Field(2, env='CHEBYSHEV_DISTANCE')
    llm_model: str = Field("gpt-4", env='LLM_MODEL')
    llm_max_tokens: int = Field(150, env='LLM_MAX_TOKENS')
    llm_temperature: float = Field(0.7, env='LLM_TEMPERATURE')
    request_delay: float = Field(0.1, env='REQUEST_DELAY')
    max_concurrent_requests: int = Field(10, env='MAX_CONCURRENT_REQUESTS')
    grid_description: str = Field("A vast grid world.", env='GRID_DESCRIPTION')
    default_message_generation_prompt: str = Field("Generate a message...", env='DEFAULT_MESSAGE_GENERATION_PROMPT')
    default_memory_generation_prompt: str = Field("Update memory...", env='DEFAULT_MEMORY_GENERATION_PROMPT')
    default_movement_generation_prompt: str = Field("Decide movement...", env='DEFAULT_MOVEMENT_GENERATION_PROMPT')
    log_queue_maxlen: int = Field(200, env='LOG_QUEUE_MAXLEN')

    class Config:
        env_file = ".env"
        env_prefix = ""

config = AppConfig()
load_dotenv()

# -------------------------------
# Pydantic Models Definitions
# -------------------------------

# Simulation Settings Model with Validators
class SimulationSettings(BaseModel):
    grid_size: int
    num_entities: int
    max_steps: int
    chebyshev_distance: int
    llm_model: str
    llm_max_tokens: int
    llm_temperature: float
    request_delay: float
    max_concurrent_requests: int

    @validator('num_entities')
    def check_num_entities(cls, v, values):
        grid_size = values.get('grid_size', 1)
        if v > grid_size ** 2:
            raise ValueError("Number of entities cannot exceed grid_size squared.")
        return v

    @validator('llm_temperature')
    def temperature_range(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("LLM temperature must be between 0 and 1.")
        return v

# Prompt Settings Model
class PromptSettings(BaseModel):
    message_generation_prompt: str
    memory_generation_prompt: str
    movement_generation_prompt: str

# Fetched Prompts Model with Validation
class FetchedPrompts(BaseModel):
    message_generation_prompt: str
    memory_generation_prompt: str
    movement_generation_prompt: str

    @validator('*')
    def validate_placeholders(cls, v):
        required_placeholders = ["{entityId}", "{x}", "{y}"]
        missing = [ph for ph in required_placeholders if ph not in v]
        if missing:
            raise ValueError(f"Missing placeholders in template: {', '.join(missing)}")
        return v

# Base Entity Model for Reusability
class BaseEntity(BaseModel):
    id: int
    name: str
    x: int
    y: int
    memory: Optional[str] = ""

# Redis Entity Model
class RedisEntity(BaseEntity):
    pass

# Supabase Entity Model
class SupabaseEntity(BaseEntity):
    pass

# LLM Response Models
class MessageResponse(BaseModel):
    message: str

class MemoryResponse(BaseModel):
    memory: str

class MovementResponse(BaseModel):
    movement: str

# Combined LLM Response using Discriminated Unions
class LLMResponse(BaseModel):
    message: Optional[str] = ""
    memory: Optional[str] = ""
    movement: Optional[str] = "stay"

# Log Entry Model
class LogEntry(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message: str

# WebSocket Event Models
class LogEvent(BaseModel):
    event_type: str = Field("log", const=True)
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ErrorEvent(BaseModel):
    event_type: str = Field("error", const=True)
    error: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class NotificationEvent(BaseModel):
    event_type: str = Field("notification", const=True)
    notification: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

WebSocketEvent = Union[LogEvent, ErrorEvent, NotificationEvent]

# Simulation State Model with Validators
class SimulationState(BaseModel):
    grid_size: int
    entities: List[BaseEntity]
    stop_signal: bool = False

    @validator("entities")
    def validate_entities(cls, entities, values):
        grid_size = values.get("grid_size", 1)
        if len(entities) > grid_size ** 2:
            raise ValueError("Too many entities for the grid size.")
        return entities

# Step Request Model
class StepRequest(BaseModel):
    steps: int

# Response Models
class EntityResponse(BaseModel):
    status: str
    entity: BaseEntity

class SyncResponse(BaseModel):
    status: str
    synchronized_entities: int

class StepResponse(BaseModel):
    status: str
    steps_performed: int

# -------------------------------
# Redis & Supabase Initialization
# -------------------------------

def get_redis_client():
    return Redis(
        host=config.redis_endpoint,
        port=6379,
        password=config.redis_password,
        decode_responses=True,
        ssl=True  # Enable SSL/TLS
    )

def get_supabase_client():
    return create_client(config.supabase_url, config.supabase_key)

redis_client = get_redis_client()
supabase_client = get_supabase_client()

# -------------------------------
# Logging Configuration
# -------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simulation_app")

# Semaphore for throttling concurrent requests
global_request_semaphore = Semaphore(config.max_concurrent_requests)

# Global log queue for real-time logging
LOG_QUEUE = deque(maxlen=config.log_queue_maxlen)  # Keeps the last 200 log messages

# Stop signal
stop_signal = False

# -------------------------------
# PydanticAI Agent Initialization
# -------------------------------

groq_model = GroqModel(
    model_name=config.llm_model,
    api_key=config.groq_api_key
)

agent = Agent(
    model=groq_model,
    result_type=LLMResponse
)

# -------------------------------
# Helper Functions
# -------------------------------

def chebyshev_distance(x1, y1, x2, y2):
    return max(abs(x1 - x2), abs(y1 - y2))

def construct_prompt(template: str, entity: dict, messages: List[str]) -> str:
    messages_str = "\n".join(messages) if messages else "No recent messages."
    memory = entity.get("memory", "No prior memory.")
    return template.format(
        entityId=entity["id"], x=entity["x"], y=entity["y"],
        grid_description=config.grid_description, memory=memory,
        messages=messages_str, distance=config.chebyshev_distance
    )

def add_log(message: str, event_type: str = "log"):
    if event_type == "log":
        entry = LogEvent(message=message)
    elif event_type == "error":
        entry = ErrorEvent(error=message)
    elif event_type == "notification":
        entry = NotificationEvent(notification=message)
    else:
        entry = LogEvent(message=message)  # Default to log

    LOG_QUEUE.append(entry.json())  # Store as JSON for consistency
    logger.info(entry.json())  # Log to the standard logger as well

async def fetch_prompts_from_fastapi() -> FetchedPrompts:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(config.groq_api_endpoint + "/prompts")
            response.raise_for_status()
            prompts = FetchedPrompts(**response.json())  # Validate the fetched prompts
            add_log("Fetched prompt templates successfully.")
            return prompts
        except httpx.HTTPError as e:
            logger.warning("Failed to fetch prompts, using default ones.")
            add_log("Failed to fetch prompts, using default ones.", event_type="error")
            return FetchedPrompts(
                message_generation_prompt=config.default_message_generation_prompt,
                memory_generation_prompt=config.default_memory_generation_prompt,
                movement_generation_prompt=config.default_movement_generation_prompt
            )

async def send_llm_request(prompt: str) -> dict:
    global stop_signal
    async with global_request_semaphore:
        if stop_signal:
            logger.info("Stopping LLM request due to stop signal.")
            add_log("Stopping LLM request due to stop signal.", event_type="notification")
            return {"message": "", "memory": "", "movement": "stay"}

        try:
            # Validate prompt
            prompt_request = PromptSettings(
                message_generation_prompt=prompt,
                memory_generation_prompt=prompt,
                movement_generation_prompt=prompt
            )  # Assuming the prompt is used for all; adjust as needed

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
            add_log(f"Error during LLM request: {e}", event_type="error")
            return {"message": "", "memory": "", "movement": "stay"}

async def initialize_entities() -> List[BaseEntity]:
    logger.info("Resetting simulation state.")
    await redis_client.flushdb()  # Clear all Redis data, including entity_keys
    add_log("Redis database flushed successfully.")

    entities = [
        BaseEntity(
            id=i,
            name=f"Entity-{i}",
            x=random.randint(0, config.grid_size - 1),
            y=random.randint(0, config.grid_size - 1),
            memory=""
        )
        for i in range(config.num_entities)
    ]

    for entity in entities:
        entity_key = f"entity:{entity.id}"
        await redis_client.hset(entity_key, mapping=entity.dict(by_alias=True))
        await redis_client.lpush("entity_keys", entity_key)  # Add to entity_keys list
        await redis_client.delete(f"entity:{entity.id}:messages")  # Clear message queue

    logger.info("Entities initialized.")
    add_log("Entities initialized successfully.", event_type="notification")
    return entities

async def fetch_nearby_messages(entity: BaseEntity, entities: List[BaseEntity], message_to_send: Optional[str] = None) -> List[str]:
    nearby_entities = [
        a for a in entities if a.id != entity.id and chebyshev_distance(entity.x, entity.y, a.x, a.y) <= config.chebyshev_distance
    ]
    received_messages = []

    for nearby_entity in nearby_entities:
        # Fetch existing messages from the nearby entity
        msg = await redis_client.hget(f"entity:{nearby_entity.id}", "message")
        logger.info(f"Fetched message for entity {nearby_entity.id}: {msg}")
        if msg:
            received_messages.append(msg)

        # If a message is being sent, add it to the recipient's queue
        if message_to_send:
            recipient_key = f"entity:{nearby_entity.id}:messages"
            await redis_client.lpush(recipient_key, f"From Entity {entity.id}: {message_to_send}")
            logger.info(f"Sent message from Entity {entity.id} to Entity {nearby_entity.id}")
            add_log(f"Sent message from Entity {entity.id} to Entity {nearby_entity.id}", event_type="notification")

    return received_messages

# -------------------------------
# Lifespan Context Manager
# -------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global stop_signal
    logger.info("Starting application lifespan...")
    try:
        # Startup logic
        await redis_client.ping()
        logger.info("Redis connection established.")
        add_log("Redis connection established.", event_type="notification")
        yield  # Application is running
    finally:
        # Shutdown logic
        logger.info("Shutting down application...")
        add_log("Shutting down application...", event_type="notification")
        stop_signal = True  # Ensure simulation stops if running
        await redis_client.close()
        logger.info("Redis connection closed.")
        add_log("Redis connection closed.", event_type="notification")

# -------------------------------
# FastAPI Application Setup
# -------------------------------

app = FastAPI(
    title="DevAGI Designer", 
    version="0.0.3", 
    description="API for world simulation and managing WebSocket logs.",
    docs_url=None, 
    lifespan=lifespan
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------------------
# Custom Swagger UI with WebSocket Logs
# -------------------------------

@app.get("/docs", tags=["Simulation"], include_in_schema=False)
async def custom_swagger_ui_html():
    logger.info("Custom /docs endpoint is being served")  # Logs when /docs is accessed
    add_log("Custom /docs endpoint accessed.", event_type="notification")
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
                    <span>MetaStrata Logs</span>
                    <span style="font-size: 18px;">&#x25B2;</span>
                </button>
            </div>
            <div id="websocket-logs-container" class="opblock-body" style="display: block; padding: 10px; background-color: #f6f6f6; border: 1px solid #e8e8e8; border-radius: 4px;">
                <div id="websocket-controls" style="text-align: center; margin-bottom: 10px;">
                    <button id="clear-logs" style="margin-right: 10px; padding: 5px 10px; border-radius: 5px; background-color: #f44336; color: white; border: none; cursor: pointer;">
                        Clear Logs
                    </button>
                    <button id="select-all" style="padding: 5px 10px; border-radius: 5px; background-color: #4caf50; color: white; border: none; cursor: pointer;">
                        Select All
                    </button>
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
            });
        </script>
    '''
    modified_html = html.body.decode("utf-8").replace("</body>", f"{custom_script}{log_area}</body>")

    return HTMLResponse(modified_html)

# -------------------------------
# Exception Handlers
# -------------------------------

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request, exc):
    add_log(f"HTTPException occurred: {exc.detail}", event_type="error")
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": f"Custom Error: {exc.detail}"}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    # Log the exception details for debugging
    logger.error(f"Unhandled exception: {str(exc)}\n{traceback.format_exc()}")
    add_log(f"Unhandled exception: {str(exc)}", event_type="error")

    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred. Please try again later."}
    )

# -------------------------------
# Simulation API Endpoints
# -------------------------------

@app.post("/reset", response_model=SyncResponse, tags=["World Simulation"])
async def reset_simulation(
    redis: Redis = Depends(get_redis_client),
    supabase: Client = Depends(get_supabase_client)
):
    global stop_signal
    try:
        # Log reset initiation
        add_log("Reset simulation process initiated.")
        stop_signal = False  # Reset stop signal before starting
        add_log("Stop signal reset to False.")

        # Clear Redis database
        await redis.flushdb()
        add_log("Redis database flushed successfully.")

        # Initialize entities
        entities = await initialize_entities()
        add_log(f"Entities reinitialized successfully. Total entities: {len(entities)}")

        # Log successful reset
        add_log("Simulation reset completed successfully.")
        return SyncResponse(status="Simulation reset successfully.", synchronized_entities=len(entities))
    except Exception as e:
        # Log any error encountered during reset
        error_message = f"Error during simulation reset: {str(e)}"
        add_log(error_message, event_type="error")
        raise HTTPException(status_code=500, detail=error_message)

@app.post("/initialize", response_model=SyncResponse, tags=["World Simulation"])
async def initialize_simulation(
    redis: Redis = Depends(get_redis_client),
    supabase: Client = Depends(get_supabase_client)
):
    global stop_signal
    try:
        # Log the initiation of the simulation
        add_log("Simulation initialization process started.")
        stop_signal = False  # Reset stop signal before starting
        add_log("Stop signal reset to False.")

        # Initialize Entities
        entities = await initialize_entities()
        add_log(f"Entities initialized successfully. Total Entities: {len(entities)}")

        # Log success and return the response
        add_log("Simulation started successfully.")
        return SyncResponse(status="Simulation started successfully.", synchronized_entities=len(entities))
    except Exception as e:
        # Log and raise an error if the initialization fails
        error_message = f"Error during simulation initialization: {str(e)}"
        add_log(error_message, event_type="error")
        raise HTTPException(status_code=500, detail=error_message)

@app.post("/step", response_model=StepResponse, tags=["World Simulation"])
async def perform_steps(
    request: StepRequest,
    redis: Redis = Depends(get_redis_client),
    supabase: Client = Depends(get_supabase_client)
):
    global stop_signal
    stop_signal = False  # Reset stop signal before starting steps

    try:
        add_log(f"Simulation steps requested: {request.steps} step(s).")

        # Fetch the current prompt templates from FastAPI
        prompts = await fetch_prompts_from_fastapi()

        logger.info("Starting simulation steps.")
        add_log("Starting simulation steps.", event_type="notification")

        for step in range(request.steps):
            if stop_signal:
                add_log("Simulation steps halted by stop signal.", event_type="notification")
                break

            # Fetch all entity keys dynamically from Redis
            entity_keys = await redis.keys("entity:*")  # Match all entity keys
            if not entity_keys:
                add_log("No entities found in Redis. Aborting simulation steps.", event_type="error")
                return StepResponse(status="No entities to process.", steps_performed=step)

            logger.info(f"Step {step + 1}: Found {len(entity_keys)} entities.")
            add_log(f"Step {step + 1}: Found {len(entity_keys)} entities.", event_type="notification")

            # Filter keys to ensure only valid hashes are processed
            valid_entity_keys = []
            for key in entity_keys:
                key_type = await redis.type(key)
                if key_type == "hash":
                    valid_entity_keys.append(key)
                else:
                    add_log(f"Skipping invalid key {key} of type {key_type}", event_type="error")

            # Fetch entity data from Redis for all valid keys
            entities = []
            redis_tasks = [redis.hgetall(key) for key in valid_entity_keys]
            entities_data = await asyncio.gather(*redis_tasks)
            for entity_data in entities_data:
                if entity_data:
                    try:
                        entity = BaseEntity(
                            id=int(entity_data["id"]),
                            name=entity_data["name"],
                            x=int(entity_data["x"]),
                            y=int(entity_data["y"]),
                            memory=entity_data.get("memory", "")
                        )
                        entities.append(entity)
                    except Exception as e:
                        add_log(f"Error parsing entity data: {str(e)}", event_type="error")

            logger.info(f"Processing {len(entities)} entities.")
            add_log(f"Processing {len(entities)} entities for step {step + 1}.", event_type="notification")

            # Process incoming messages for each entity
            for entity in entities:
                try:
                    # Fetch the existing message field
                    message = await redis.hget(f"entity:{entity.id}", "message")

                    if message:
                        logger.info(f"Entity {entity.id} received message: {message}")
                        add_log(f"Entity {entity.id} received message: {message}", event_type="notification")

                        # Optionally update memory or trigger actions based on the message
                        updated_memory = f"{entity.memory} | Received: {message}"
                        await redis.hset(f"entity:{entity.id}", "memory", updated_memory)

                        # Clear the message field after processing (if required)
                        await redis.hset(f"entity:{entity.id}", "message", "")
                except Exception as e:
                    logger.error(f"Error processing message for Entity {entity.id}: {str(e)}")
                    add_log(f"Error processing message for Entity {entity.id}: {str(e)}", event_type="error")

            # Clear message queues only after processing all entities
            for entity in entities:
                await redis.delete(f"entity:{entity.id}:messages")

            # Message Generation
            for entity in entities:
                try:
                    messages = await fetch_nearby_messages(entity, entities)
                    message_prompt = prompts.message_generation_prompt
                    message_result = await send_llm_request(
                        construct_prompt(message_prompt, entity.dict(), messages)
                    )
                    if "message" in message_result and message_result["message"]:
                        await redis.hset(f"entity:{entity.id}", "message", message_result["message"])
                        add_log(f"Message generated for Entity {entity.id}: {message_result['message']}", event_type="notification")
                except Exception as e:
                    logger.error(f"Error generating message for Entity {entity.id}: {str(e)}")
                    add_log(f"Error generating message for Entity {entity.id}: {str(e)}", event_type="error")
                await asyncio.sleep(config.request_delay)

            if stop_signal:
                logger.info("Stopping after message generation due to stop signal.")
                add_log("Simulation steps halted after message generation by stop signal.", event_type="notification")
                break

            # Memory Generation
            for entity in entities:
                try:
                    messages = await fetch_nearby_messages(entity, entities)
                    memory_prompt = prompts.memory_generation_prompt
                    memory_result = await send_llm_request(
                        construct_prompt(memory_prompt, entity.dict(), messages)
                    )
                    if "memory" in memory_result and memory_result["memory"]:
                        await redis.hset(f"entity:{entity.id}", "memory", memory_result["memory"])
                        add_log(f"Memory updated for Entity {entity.id}: {memory_result['memory']}", event_type="notification")
                except Exception as e:
                    logger.error(f"Error generating memory for Entity {entity.id}: {str(e)}")
                    add_log(f"Error generating memory for Entity {entity.id}: {str(e)}", event_type="error")
                await asyncio.sleep(config.request_delay)

            if stop_signal:
                logger.info("Stopping after memory generation due to stop signal.")
                add_log("Simulation steps halted after memory generation by stop signal.", event_type="notification")
                break

            # Movement Generation
            for entity in entities:
                try:
                    movement_prompt = prompts.movement_generation_prompt
                    movement_result = await send_llm_request(
                        construct_prompt(movement_prompt, entity.dict(), [])
                    )
                    if "movement" in movement_result and movement_result["movement"]:
                        movement = movement_result["movement"].strip().lower()
                        initial_position = (entity.x, entity.y)

                        # Apply movement logic
                        if movement == "x+1":
                            entity.x = (entity.x + 1) % config.grid_size
                        elif movement == "x-1":
                            entity.x = (entity.x - 1) % config.grid_size
                        elif movement == "y+1":
                            entity.y = (entity.y + 1) % config.grid_size
                        elif movement == "y-1":
                            entity.y = (entity.y - 1) % config.grid_size
                        elif movement == "stay":
                            logger.info(f"Entity {entity.id} stays in place at {initial_position}.")
                            add_log(f"Entity {entity.id} stays in place at {initial_position}.", event_type="notification")
                            continue
                        else:
                            logger.warning(f"Invalid movement command for Entity {entity.id}: {movement}")
                            add_log(f"Invalid movement command for Entity {entity.id}: {movement}", event_type="error")
                            continue

                        # Log and update position
                        logger.info(f"Entity {entity.id} moved from {initial_position} to ({entity.x}, {entity.y}) with action '{movement}'.")
                        add_log(f"Entity {entity.id} moved from {initial_position} to ({entity.x}, {entity.y}) with action '{movement}'.", event_type="notification")
                        await redis.hset(f"entity:{entity.id}", mapping={"x": entity.x, "y": entity.y})
                except Exception as e:
                    logger.error(f"Error generating movement for Entity {entity.id}: {str(e)}")
                    add_log(f"Error generating movement for Entity {entity.id}: {str(e)}", event_type="error")
                await asyncio.sleep(config.request_delay)

        logger.info(f"Completed {request.steps} step(s).")
        add_log(f"Simulation steps completed: {request.steps} step(s).", event_type="notification")
        return StepResponse(status=f"Performed {request.steps} step(s).", steps_performed=request.steps)

@app.post("/stop", tags=["World Simulation"])
async def stop_simulation():
    global stop_signal
    try:
        # Log the start of the stop process
        add_log("Stop simulation process initiated.", event_type="notification")

        # Set the stop signal
        stop_signal = True
        add_log("Stop signal triggered successfully.", event_type="notification")

        # Log successful completion
        add_log("Simulation stopping process completed.", event_type="notification")
        return {"status": "Simulation stopping."}
    except Exception as e:
        # Log and raise any unexpected errors
        error_message = f"Error during simulation stop process: {str(e)}"
        add_log(error_message, event_type="error")
        raise HTTPException(status_code=500, detail=error_message)

# -------------------------------
# Entity Management Endpoints
# -------------------------------

@app.post("/entities", response_model=EntityResponse, tags=["Entities"])
async def create_entity(
    entity: BaseEntity,
    redis: Redis = Depends(get_redis_client),
    supabase: Client = Depends(get_supabase_client)
):
    entity_key = f"entity:{entity.id}"

    try:
        # Log the attempt to create a new entity
        add_log(f"Attempting to create new entity with ID {entity.id} and name '{entity.name}'.", event_type="notification")

        # Check if the ID already exists in Redis
        if await redis.exists(entity_key):
            error_message = f"Entity ID {entity.id} already exists in Redis."
            add_log(error_message, event_type="error")
            raise HTTPException(status_code=400, detail=error_message)

        # Check if the ID already exists in Supabase
        existing_entity = supabase.table("entities").select("id").eq("id", entity.id).execute()
        if existing_entity.data:
            error_message = f"Entity ID {entity.id} already exists in Supabase."
            add_log(error_message, event_type="error")
            raise HTTPException(status_code=400, detail=error_message)

        # Save entity data in Redis
        await redis.hset(entity_key, mapping=entity.dict(by_alias=True))
        add_log(f"Entity data for ID {entity.id} saved in Redis.", event_type="notification")

        # Add the entity key to the Redis list
        await redis.lpush("entity_keys", entity_key)
        add_log(f"Entity key for ID {entity.id} added to Redis entity_keys list.", event_type="notification")

        # Save the entity in Supabase
        supabase.table("entities").insert(entity.dict()).execute()
        add_log(f"Entity data for ID {entity.id} saved in Supabase.", event_type="notification")

        # Log the successful creation of the entity
        add_log(f"New entity created successfully: ID={entity.id}, Name={entity.name}, Position=({entity.x}, {entity.y})", event_type="notification")
        
        return EntityResponse(status="Entity created successfully.", entity=entity)

    except Exception as e:
        # Log and raise unexpected errors
        error_message = f"Error creating entity with ID {entity.id}: {str(e)}"
        add_log(error_message, event_type="error")
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/entities/{entity_id}", response_model=BaseEntity, tags=["Entities"])
async def get_entity(
    entity_id: int,
    redis: Redis = Depends(get_redis_client)
):
    # Log the attempt to fetch entity data
    add_log(f"Fetching data for entity with ID {entity_id}.", event_type="notification")

    # Fetch entity data from Redis
    entity_data = await redis.hgetall(f"entity:{entity_id}")
    if not entity_data:
        error_message = f"Entity with ID {entity_id} not found."
        add_log(error_message, event_type="error")
        raise HTTPException(status_code=404, detail=error_message)

    # Convert Redis data into a BaseEntity model (ensure proper types are cast)
    try:
        entity = BaseEntity(
            id=entity_id,
            name=entity_data.get("name"),
            x=int(entity_data.get("x")),
            y=int(entity_data.get("y")),
            memory=entity_data.get("memory", "")
        )
        add_log(f"Successfully fetched entity with ID {entity_id}, Name: {entity.name}, Position: ({entity.x}, {entity.y}).", event_type="notification")
        return entity
    except Exception as e:
        error_message = f"Failed to parse data for entity ID {entity_id}: {str(e)}"
        add_log(error_message, event_type="error")
        raise HTTPException(status_code=500, detail=error_message)

@app.put("/entities/{entity_id}", response_model=BaseEntity, tags=["Entities"])
async def update_entity(
    entity_id: int,
    entity: BaseEntity,
    redis: Redis = Depends(get_redis_client),
    supabase: Client = Depends(get_supabase_client)
):
    try:
        # Log the attempt to update entity data
        add_log(f"Attempting to update entity with ID {entity_id}.", event_type="notification")

        # Update entity data in Redis
        await redis.hset(f"entity:{entity_id}", mapping=entity.dict(by_alias=True))
        add_log(f"Entity with ID {entity_id} updated in Redis.", event_type="notification")

        # Update entity in Supabase
        supabase.table("entities").update(entity.dict()).eq("id", entity_id).execute()
        add_log(f"Entity with ID {entity_id} updated in Supabase.", event_type="notification")

        # Log successful update
        add_log(f"Successfully updated entity with ID {entity_id}.", event_type="notification")
        return entity
    except Exception as e:
        # Log and raise error if update fails
        error_message = f"Error updating entity with ID {entity_id}: {str(e)}"
        add_log(error_message, event_type="error")
        raise HTTPException(status_code=500, detail=error_message)

@app.delete("/entities/{entity_id}", tags=["Entities"])
async def delete_entity(
    entity_id: int,
    redis: Redis = Depends(get_redis_client),
    supabase: Client = Depends(get_supabase_client)
):
    entity_key = f"entity:{entity_id}"
    try:
        # Log the attempt to delete an entity
        add_log(f"Attempting to delete entity with ID {entity_id}.", event_type="notification")

        # Delete entity from Redis
        await redis.delete(entity_key)
        add_log(f"Entity with ID {entity_id} deleted from Redis.", event_type="notification")

        # Remove the key from the Redis list
        await redis.lrem("entity_keys", 0, entity_key)
        add_log(f"Entity key with ID {entity_id} removed from Redis entity_keys list.", event_type="notification")

        # Optionally, delete the entity from Supabase
        supabase.table("entities").delete().eq("id", entity_id).execute()
        add_log(f"Entity with ID {entity_id} deleted from Supabase.", event_type="notification")

        return {"status": "Entity deleted successfully"}
    except Exception as e:
        # Log and raise any unexpected errors
        error_message = f"Error deleting entity with ID {entity_id}: {str(e)}"
        add_log(error_message, event_type="error")
        raise HTTPException(status_code=500, detail=error_message)

# -------------------------------
# Utilities Endpoints
# -------------------------------

@app.post("/entities/{recipient_id}/create_memory", tags=["Utilities"])
async def create_memory(
    recipient_id: int,
    message: str = Query(..., description="The memory content to add or update for the recipient entity."),
    redis: Redis = Depends(get_redis_client)
):
    recipient_key = f"entity:{recipient_id}"

    try:
        # Log the attempt to create memory
        add_log(f"Creating a memory for Entity {recipient_id}: \"{message}\".", event_type="notification")

        # Validate that the recipient exists
        if not await redis.exists(recipient_key):
            error_message = f"Recipient Entity ID {recipient_id} not found."
            add_log(error_message, event_type="error")
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
        add_log(f"Memory updated successfully for Entity {recipient_id}: \"{message}\".", event_type="notification")

        return {"status": "Memory updated successfully", "message": message}

    except Exception as e:
        # Log and raise any unexpected errors
        error_message = f"Error creating memory for Entity {recipient_id}: {str(e)}"
        add_log(error_message, event_type="error")
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/entities/{entity_id}/nearby", response_model=List[BaseEntity], tags=["Utilities"])
async def get_nearby_entities(
    entity_id: int,
    redis: Redis = Depends(get_redis_client)
):
    try:
        # Log the attempt to fetch nearby entities
        add_log(f"Fetching nearby entities for Entity ID {entity_id}.", event_type="notification")

        # Get the entity's position from Redis
        entity_data = await redis.hgetall(f"entity:{entity_id}")
        if not entity_data:
            error_message = f"Entity with ID {entity_id} not found."
            add_log(error_message, event_type="error")
            raise HTTPException(status_code=404, detail=error_message)

        entity = BaseEntity(**entity_data)

        # Fetch all entities except the current one
        all_entity_keys = await redis.keys("entity:*")
        all_entity_keys = [key for key in all_entity_keys if key != f"entity:{entity_id}"]
        redis_tasks = [redis.hgetall(key) for key in all_entity_keys]
        entities_data = await asyncio.gather(*redis_tasks)
        all_entities = []
        for data in entities_data:
            if data:
                try:
                    ent = BaseEntity(**data)
                    all_entities.append(ent)
                except Exception as e:
                    add_log(f"Error parsing entity data: {str(e)}", event_type="error")

        # Filter nearby entities based on Chebyshev distance
        nearby_entities = [
            a for a in all_entities
            if chebyshev_distance(entity.x, entity.y, a.x, a.y) <= config.chebyshev_distance
        ]

        add_log(f"Nearby entities fetched successfully for Entity ID {entity_id}. Total nearby entities: {len(nearby_entities)}.", event_type="notification")
        return nearby_entities
    except Exception as e:
        # Log and raise any unexpected errors
        error_message = f"Error fetching nearby entities for Entity ID {entity_id}: {str(e)}"
        add_log(error_message, event_type="error")
        raise HTTPException(status_code=500, detail=error_message)

@app.post("/sync_entity", response_model=SyncResponse, tags=["Utilities"])
async def sync_entity(
    redis: Redis = Depends(get_redis_client),
    supabase: Client = Depends(get_supabase_client)
):
    try:
        # Log the start of the synchronization process
        add_log("Synchronization process initiated between Redis and Supabase.", event_type="notification")

        # Fetch all entity keys
        entity_keys = await redis.keys("entity:*")
        redis_tasks = [redis.hgetall(key) for key in entity_keys]
        entities_data = await asyncio.gather(*redis_tasks)
        all_entities = []
        for data in entities_data:
            if data:
                try:
                    entity = SupabaseEntity(**data)
                    all_entities.append(entity)
                except Exception as e:
                    add_log(f"Error parsing entity data: {str(e)}", event_type="error")

        # Synchronize each entity to Supabase
        for entity in all_entities:
            supabase.table("entities").upsert(entity.dict()).execute()
            add_log(f"Entity with ID {entity.id} synchronized to Supabase.", event_type="notification")

        add_log("Synchronization process completed successfully.", event_type="notification")
        return SyncResponse(status="Entities synchronized between Redis and Supabase", synchronized_entities=len(all_entities))
    except Exception as e:
        # Log and raise any unexpected errors
        error_message = f"Error during entity synchronization: {str(e)}"
        add_log(error_message, event_type="error")
        raise HTTPException(status_code=500, detail=error_message)

# -------------------------------
# Settings Endpoints
# -------------------------------

@app.get("/settings", response_model=SimulationSettings, tags=["Settings"])
async def get_settings(
    redis: Redis = Depends(get_redis_client)
):
    try:
        # Log the attempt to fetch simulation settings
        add_log("Fetching simulation settings.", event_type="notification")

        # Fetch all entity keys from Redis
        entity_keys = await redis.keys("entity:*")  # Get all keys matching entity pattern
        num_entities = len(entity_keys)  # Count the number of entities

        # Log successful retrieval
        add_log(f"Simulation settings fetched successfully. Total entities: {num_entities}.", event_type="notification")

        # Return the dynamically updated number of entities
        return SimulationSettings(
            grid_size=config.grid_size,
            num_entities=num_entities,
            max_steps=config.max_steps,
            chebyshev_distance=config.chebyshev_distance,
            llm_model=config.llm_model,
            llm_max_tokens=config.llm_max_tokens,
            llm_temperature=config.llm_temperature,
            request_delay=config.request_delay,
            max_concurrent_requests=config.max_concurrent_requests
        )
    except Exception as e:
        # Log and raise any unexpected errors
        error_message = f"Error fetching simulation settings: {str(e)}"
        add_log(error_message, event_type="error")
        raise HTTPException(status_code=500, detail=error_message)

@app.post("/settings", response_model=SimulationSettings, tags=["Settings"])
async def set_settings(
    settings: SimulationSettings,
    redis: Redis = Depends(get_redis_client)
):
    try:
        # Log the attempt to update simulation settings
        add_log("Updating simulation settings.", event_type="notification")

        # Update global configuration
        global config
        config.grid_size = settings.grid_size
        config.num_entities = settings.num_entities
        config.max_steps = settings.max_steps
        config.chebyshev_distance = settings.chebyshev_distance
        config.llm_model = settings.llm_model
        config.llm_max_tokens = settings.llm_max_tokens
        config.llm_temperature = settings.llm_temperature
        config.request_delay = settings.request_delay
        config.max_concurrent_requests = settings.max_concurrent_requests

        # Log successful update
        add_log("Simulation settings updated successfully.", event_type="notification")
        return settings
    except Exception as e:
        # Log and raise any unexpected errors
        error_message = f"Error updating simulation settings: {str(e)}"
        add_log(error_message, event_type="error")
        raise HTTPException(status_code=500, detail=error_message)

# -------------------------------
# Prompt Templates Endpoints
# -------------------------------

@app.get("/prompts", response_model=PromptSettings, tags=["Settings"])
async def get_prompts():
    try:
        # Log the attempt to fetch prompt templates
        add_log("Fetching current prompt templates.", event_type="notification")

        # Return the prompt templates
        prompts = PromptSettings(
            message_generation_prompt=config.default_message_generation_prompt,
            memory_generation_prompt=config.default_memory_generation_prompt,
            movement_generation_prompt=config.default_movement_generation_prompt
        )

        # Log successful retrieval
        add_log("Prompt templates fetched successfully.", event_type="notification")
        return prompts
    except Exception as e:
        # Log and raise any unexpected errors
        error_message = f"Error fetching prompt templates: {str(e)}"
        add_log(error_message, event_type="error")
        raise HTTPException(status_code=500, detail=error_message)

@app.post("/prompts", response_model=PromptSettings, tags=["Settings"])
async def set_prompts(
    prompts: PromptSettings,
    redis: Redis = Depends(get_redis_client)
):
    try:
        # Log the attempt to update prompt templates
        add_log("Updating prompt templates.", event_type="notification")

        # Update global configuration
        config.default_message_generation_prompt = prompts.message_generation_prompt
        config.default_memory_generation_prompt = prompts.memory_generation_prompt
        config.default_movement_generation_prompt = prompts.movement_generation_prompt

        # Log successful update
        add_log("Prompt templates updated successfully.", event_type="notification")
        return prompts
    except Exception as e:
        # Log and raise any unexpected errors
        error_message = f"Error updating prompt templates: {str(e)}"
        add_log(error_message, event_type="error")
        raise HTTPException(status_code=500, detail=error_message)

# -------------------------------
# WebSocket Endpoint for Real-Time Logs
# -------------------------------

@app.websocket("/logs")
async def websocket_log_endpoint(websocket: WebSocket):
    add_log("WebSocket connection initiated for real-time logs.", event_type="notification")
    await websocket.accept()
    try:
        while True:
            if LOG_QUEUE:
                # Send all current logs to the client
                for log in list(LOG_QUEUE):
                    try:
                        event = WebSocketEvent.parse_raw(log)
                        await websocket.send_json(event.dict())
                    except Exception as e:
                        logger.error(f"Error sending log via WebSocket: {str(e)}")
                        add_log(f"Error sending log via WebSocket: {str(e)}", event_type="error")
                LOG_QUEUE.clear()  # Clear the queue after sending
            await asyncio.sleep(1)  # Check for new logs every second
    except asyncio.CancelledError:
        # Log graceful cancellation
        add_log("WebSocket log stream cancelled by the client.", event_type="notification")
    except Exception as e:
        # Log unexpected exceptions
        error_message = f"Error in WebSocket log stream: {str(e)}"
        logger.error(error_message)
        add_log(error_message, event_type="error")
    finally:
        # Log connection close and clean up
        add_log("WebSocket connection for real-time logs closed.", event_type="notification")
        await websocket.close()

# -------------------------------
# Main Entry Point
# -------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
