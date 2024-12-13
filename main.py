# main.py

import random
import asyncio
import json
import httpx
import logging
import traceback
from asyncio import Lock, Semaphore
from contextlib import asynccontextmanager
from core.app import app
from datetime import datetime
from dotenv import load_dotenv
from fastapi import Body, Path, Query, HTTPException, WebSocket
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from endpoints import router
from endpoints.database import redis
from endpoints.simulation import initialize_world
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from redis.asyncio.lock import Lock
from typing import Dict, List, Optional
from utils import add_log, LOG_QUEUE, logger, submit_summary
from world import World
from config import (
    GROQ_API_KEY,
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
    create_world_config,
    CONNECTIVITY_GRAPH
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

# Initialize a dictionary to hold Redis locks for each world (0-based)
world_locks: Dict[int, Lock] = {world_id: Lock(redis, f"lock:world:{world_id}", timeout=10) for world_id in range(NUM_WORLDS)}

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

def construct_prompt(template, entity: Entity, messages):
    sanitized_messages = [msg.replace("\n", " ").replace("\"", "'").strip() for msg in messages]
    messages_str = "\n".join(sanitized_messages) if sanitized_messages else "No recent messages."
    memory = entity.memory.replace("\n", " ").replace("\"", "'")

    return template.format(
        entityId=entity.id, x=entity.x, y=entity.y,
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

# Helper function for Worlds. Randomly selects a movement action for an agent.
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

# Helper function for Worlds. Randomly generates messages between agents with a certain probability.
def generate_communications(world: World) -> List[Dict]:
    """
    Generate communications between worlds.
    """
    messages = []
    for agent in world.agents:
        if random.random() < 0.1:  # 10% chance to send a message
            # Example: Send a state summary request to a connected world
            connected_worlds = list(CONNECTIVITY_GRAPH.neighbors(world.world_id))
            if connected_worlds:
                target_world_id = random.choice(connected_worlds)
                message = {
                    "source_world_id": world.world_id,
                    "message_type": "state_summary_request",
                    "payload": {}
                }
                messages.append(message)
                # Use the messaging endpoint to send the message
                asyncio.create_task(send_world_message_async(target_world_id, message))
                add_log(f"World {world.world_id} sent 'state_summary_request' to World {target_world_id}.")
    return messages

async def send_world_message_async(target_world_id: int, message: Dict):
    """
    Asynchronously send a message to another world using the API endpoint.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://localhost:8000/worlds/{target_world_id}/send_message",
                json=message
            )
            if response.status_code == 200:
                add_log(f"Successfully sent message to World {target_world_id}.")
            else:
                add_log(f"Failed to send message to World {target_world_id}: {response.text}")
    except Exception as e:
        add_log(f"Exception occurred while sending message to World {target_world_id}: {e}")

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

async def initialize_entities():
    logger.info("Resetting simulation state.")
    await redis.flushdb()  # Clear all Redis data, including entity_keys and world-specific data

    entities = []
    entities_per_world = NUM_ENTITIES // NUM_WORLDS
    extra = NUM_ENTITIES % NUM_WORLDS  # Handle remainder

    entity_id = 0
    for world_id in range(1, NUM_WORLDS + 1):
        num_agents = entities_per_world + (1 if world_id <= extra else 0)
        for _ in range(num_agents):
            entity = {
                "id": entity_id,
                "name": f"Entity-{entity_id}",
                "x": random.randint(0, GRID_SIZE - 1),
                "y": random.randint(0, GRID_SIZE - 1),
                "memory": "",
                "world_id": world_id  # Assign to a world
            }
            entity_key = f"entity:{entity_id}"
            await redis.hset(entity_key, mapping=entity)
            await redis.lpush("entity_keys", entity_key)  # Add to entity_keys list
            await redis.delete(f"entity:{entity_id}:messages")  # Clear message queue
            entities.append(entity)
            entity_id += 1

    logger.info(f"Entities initialized and assigned to {NUM_WORLDS} worlds.")
    return entities
async def fetch_nearby_messages(entity, entities, message_to_send=None):
    """
    Fetch messages from nearby entities and optionally send a message to them.

    Args:
        entity (dict): The entity fetching nearby messages.
        entities (list): List of all entities.
        message_to_send (str, optional): Message to send to nearby entities. Defaults to None.

    Returns:
        list: A list of received messages from nearby entities.
    """
    try:
        # Identify nearby entities within the defined Chebyshev distance
        nearby_entities = [
            a for a in entities
            if a["id"] != entity["id"] and chebyshev_distance(entity["x"], entity["y"], a["x"], a["y"]) <= CHEBYSHEV_DISTANCE
        ]
        received_messages = []

        # Fetch messages from nearby entities
        for nearby_entity in nearby_entities:
            msg = await redis.hget(f"entity:{nearby_entity['id']}", "message")
            if msg:
                logger.info(f"Fetched message for entity {nearby_entity['id']}: {msg}")
                received_messages.append(msg)

            # If a message is being sent, add it to the recipient's queue
            if message_to_send:
                recipient_key = f"entity:{nearby_entity['id']}:messages"
                await redis.lpush(recipient_key, f"From Entity {entity['id']}: {message_to_send}")
                logger.info(f"Sent message from Entity {entity['id']} to Entity {nearby_entity['id']}")

        return received_messages

    except Exception as e:
        logger.error(f"Error fetching or sending messages for Entity {entity['id']}: {e}")
        return []

# World Simulation Functions

async def simulate_world_step(world: World):
    """
    Update the state of the world for one simulation step.
    This includes updating agents, tasks, resources, handling communications, processing messages, and handling memory.
    """
    try:
        # Fetch world-specific tasks and resources
        tasks = await redis.hgetall(f"world:{world.world_id}:tasks")
        resources = await redis.hgetall(f"world:{world.world_id}:resources")
        
        # Convert tasks back to list of dicts
        tasks = [json.loads(t) for t in tasks.values()]
        resources['distribution'] = json.loads(resources['distribution'])

        # Update agent positions
        for agent in world.agents:
            move = determine_agent_movement(agent)  # Implement this function
            agent["x"] = (agent["x"] + move["x"]) % world.grid_size
            agent["y"] = (agent["y"] + move["y"]) % world.grid_size

        # Update tasks
        for task in tasks:
            task["progress"] += task.get("increment", 0.1)
            task["progress"] = min(task["progress"], 1.0)  # Cap at 100%

        # Update resources
        consumed = int(resources.get("consumed", 0)) + 1  # Increment consumption
        resources["consumed"] = consumed

        # Handle communications
        messages = generate_communications(world)  # Implement this function
        world.communications.extend(messages)

        # Handle memory updates
        for agent in world.agents:
            # Fetch nearby messages for the agent
            nearby_messages = await fetch_nearby_messages(agent, world.agents)

            # Update memory based on the messages received
            if nearby_messages:
                memory_update = "\n".join(nearby_messages)
                agent["memory"] += f"\n{memory_update}"

                # Optionally, truncate memory to avoid excessive growth
                max_memory_size = 1000  # Adjust as needed
                if len(agent["memory"]) > max_memory_size:
                    agent["memory"] = agent["memory"][-max_memory_size:]

        # Receive and process incoming messages via consumer groups
        await world.receive_messages()

        # Persist updated tasks, resources, and agent states back to Redis
        pipeline = redis.pipeline()

        # Update tasks
        pipeline.delete(f"world:{world.world_id}:tasks")
        for task in tasks:
            pipeline.hset(f"world:{world.world_id}:tasks", f"task_{task['id']}", json.dumps(task))

        # Update resources
        resources['distribution'] = json.dumps(resources['distribution'])
        pipeline.hset(f"world:{world.world_id}:resources", mapping={
            "total": resources["total"],
            "consumed": resources["consumed"],
            "distribution": resources["distribution"]
        })

        # Update agents in Redis
        for agent in world.agents:
            agent_key = f"agent:{world.world_id}:{agent['id']}"
            pipeline.hset(agent_key, mapping={
                "x": agent["x"],
                "y": agent["y"],
                "memory": agent["memory"]
            })

        await pipeline.execute()

        logger.debug(f"World {world.world_id} state updated with memory.")
        add_log(f"World {world.world_id} state updated with memory.")

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

async def resolve_messages_in_worlds(worlds):
    """
    Periodically resolve and route messages between worlds.
    This function can be scheduled as a background task.
    """
    while not stop_signal:
        try:
            for world in worlds:
                if isinstance(world, World):
                    await world.receive_messages()
                    # Removed: world.process_messages()
        except Exception as e:
            logger.error(f"Error in message resolution loop: {e}")
        await asyncio.sleep(1)  # Adjust interval as needed

# Initialize connectivity dynamically
def initialize_connectivity(num_worlds: int):
    for world_id in range(num_worlds):
        # Define potential target world IDs excluding the current world_id
        potential_targets = list(range(num_worlds))
        potential_targets.remove(world_id)

        # Randomly select 1 to 3 connections
        num_connections = random.randint(1, 3)
        connections = random.sample(potential_targets, min(num_connections, len(potential_targets)))

        # Add edges to the connectivity graph
        for target_id in connections:
            CONNECTIVITY_GRAPH.add_edge(world_id, target_id)

        add_log(f"World {world_id} connected to Worlds {connections}.")

async def get_entities_by_world(world_id: int):
    """
    Retrieve all entities associated with a specific world.
    """
    pattern = "entity:*"
    entity_keys = await redis.keys(pattern)
    entities = []
    for key in entity_keys:
        entity_data = await redis.hgetall(key)
        if entity_data and int(entity_data.get("world_id", -1)) == world_id:
            entity = Entity(
                id=int(entity_data.get("id", 0)),
                name=entity_data.get("name", ""),
                x=int(entity_data.get("x", 0)),
                y=int(entity_data.get("y", 0)),
                memory=entity_data.get("memory", ""),
            )
            entities.append(entity)
    return entities

@asynccontextmanager
async def lifespan(app):
    global stop_signal
    logger.info("Starting application lifespan...")

    # Ensure `worlds` attribute exists
    if not hasattr(app.state, "worlds"):
        app.state.worlds = []

    try:
        # Redis initialization
        await redis.ping()
        logger.info("Redis connection established.")

        # Adjust worlds based on NUM_WORLDS
        current_count = len(app.state.worlds)
        if NUM_WORLDS > current_count:
            for i in range(current_count, NUM_WORLDS):
                world_id = i  # 0-based indexing

                # Initialize tasks
                tasks = [
                    {
                        "id": t,
                        "progress": 0.0,
                        "duration": random.randint(5, 20),
                        "increment": 0.1,
                        "priority": 1.0,
                    }
                    for t in range(100)
                ]

                # Initialize resources
                resources = {
                    "total": 1000,
                    "consumed": 0,
                    "distribution": {agent_id: 100 for agent_id in range(NUM_ENTITIES)},
                }

                # Create world configuration
                world_config = create_world_config(
                    world_id=world_id,
                    grid_size=GRID_SIZE,
                    num_agents=NUM_ENTITIES,
                    tasks=tasks,
                    resources=resources,
                )

                # Create World instance
                world = World(**world_config)
                app.state.worlds.append(world)
                logger.info(f"World {world.world_id} initialized with {world.num_agents} agents.")

                # Persist world state to Redis under separate namespaces
                await initialize_world(world.world_id)
                logger.info(f"World {world.world_id} persisted to Redis.")

        elif NUM_WORLDS < current_count:
            # Optionally handle world removal
            app.state.worlds = app.state.worlds[:NUM_WORLDS]

        logger.info(f"Adjusted to {NUM_WORLDS} worlds")

        # Initialize dynamic connectivity graph
        initialize_connectivity(NUM_WORLDS)
        logger.info("Dynamic connectivity graph initialized.")

        # Start background task for message resolution
        asyncio.create_task(resolve_messages_in_worlds(app.state.worlds))

        yield  # Application runs here

    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)

    finally:
        logger.info("Shutting down application...")
        stop_signal = True
        try:
            await redis.close()
            logger.info("Redis connection closed.")
        except Exception as e:
            logger.error(f"Error while closing Redis: {e}", exc_info=True)

# Attach lifespan context manager
app.router.lifespan_context = lifespan

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

            // WebSocket connection for real-time logs
            let websocket;

            function connectWebSocket() {
                websocket = new WebSocket("ws://localhost:8000/logs");
                
                websocket.onopen = function(event) {
                    addLog("Connected to WebSocket for real-time logs.");
                };

                websocket.onmessage = function(event) {
                    const logsDiv = document.getElementById("websocket-logs");
                    const logEntry = document.createElement("p");
                    logEntry.textContent = event.data;
                    logsDiv.appendChild(logEntry);
                    logsDiv.scrollTop = logsDiv.scrollHeight; // Auto-scroll to the bottom
                };

                websocket.onclose = function(event) {
                    addLog("WebSocket connection closed.");
                };

                websocket.onerror = function(error) {
                    addLog("WebSocket error: " + error.message);
                };
            }

            function addLog(message) {
                const logsDiv = document.getElementById("websocket-logs");
                const logEntry = document.createElement("p");
                logEntry.textContent = message;
                logsDiv.appendChild(logEntry);
                logsDiv.scrollTop = logsDiv.scrollHeight; // Auto-scroll to the bottom
            }

            // Connect on page load
            document.addEventListener("DOMContentLoaded", function() {
                connectWebSocket();
            });

            // Reconnect button
            const reconnectButton = document.getElementById("reconnect-stream");
            if (reconnectButton) {
                reconnectButton.addEventListener("click", function() {
                    if (websocket.readyState === WebSocket.OPEN) {
                        websocket.close();
                    }
                    connectWebSocket();
                    addLog("Attempting to reconnect WebSocket...");
                });
            }

            // Copy logs button
            const copyLogsButton = document.getElementById("copy-logs");
            if (copyLogsButton) {
                copyLogsButton.addEventListener("click", function() {
                    const logsDiv = document.getElementById("websocket-logs");
                    const text = logsDiv.innerText;
                    navigator.clipboard.writeText(text).then(() => {
                        addLog("Logs copied to clipboard.");
                    }).catch(err => {
                        addLog("Failed to copy logs: " + err);
                    });
                });
            }
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

@app.post("/worlds/{world_id}/step", tags=["Simulation"])
async def perform_steps_for_world(
    world_id: int = Path(..., description="ID of the world to perform steps for"),
    request: StepRequest = Body(...)
):
    """
    Perform simulation steps for a specific world.

    - **world_id**: ID of the world to perform steps for.
    - **steps**: Number of simulation steps to perform.
    """
    # Validate the number of steps
    if request.steps <= 0:
        error_message = "Number of steps must be a positive integer."
        add_log(error_message)
        raise HTTPException(status_code=400, detail=error_message)

    # Retrieve the lock for the specified world_id
    lock = world_locks.get(world_id)
    if not lock:
        error_message = f"No lock found for World {world_id}."
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

    # Acquire the Redis lock to prevent concurrent step executions
    async with lock:
        global stop_signal
        stop_signal = False  # Reset stop signal before starting steps

        try:
            add_log(f"Simulation steps requested: {request.steps} step(s) for World {world_id}.")

            # Fetch the current prompt templates from FastAPI
            prompts = await fetch_prompts_from_fastapi()
            add_log("Fetched prompt templates successfully.")

            logger.info(f"Starting simulation steps for World {world_id}...")

            # Locate the specific world
            world = next(
                (w for w in app.state.worlds if isinstance(w, World) and w.world_id == world_id),
                None
            )
            if not world:
                error_message = f"World with ID {world_id} not found."
                add_log(error_message)
                raise HTTPException(status_code=404, detail=error_message)

            for step in range(request.steps):
                if stop_signal:
                    add_log("Simulation steps halted by stop signal.")
                    break

                # Perform simulation logic for the specific world
                try:
                    # Simulate world step
                    await simulate_world_step(world)

                    # Summarize state and submit to mTNN
                    summary_vector = world.summarize_state()
                    await submit_summary(world.world_id, summary_vector)

                    # Log the summary
                    add_log(f"World {world.world_id} summary: {summary_vector}")
                except Exception as e:
                    logger.error(f"Error processing World {world.world_id}: {str(e)}")
                    add_log(f"Error processing World {world.world_id}: {str(e)}")

                # Fetch entities for the specific world
                entities = await get_entities_by_world(world.world_id)
                logger.info(f"World {world.world_id}: Found {len(entities)} entities.")
                add_log(f"World {world.world_id}: Found {len(entities)} entities.")

                # Process incoming messages for each entity
                for entity in entities:
                    try:
                        # Fetch the existing message field
                        message = await redis.hget(f"entity:{nearby_entity['id']}", "message")

                        if message:
                            logger.info(f"Entity {entity.id} received message: {message}")
                            add_log(f"Entity {entity.id} received message: {message}")

                            # Optionally update memory or trigger actions based on the message
                            updated_memory = f"{entity.memory}\nReceived: {message}"
                            await redis.hset(f"entity:{entity.id}", "memory", updated_memory)

                            # Clear the message field after processing (if required)
                            await redis.hset(f"entity:{entity.id}", "message", "")
                    except Exception as e:
                        logger.error(f"Error processing message for Entity {entity.id}: {str(e)}")
                        add_log(f"Error processing message for Entity {entity.id}: {str(e)}")

                # Clear message queues only after processing all entities
                for entity in entities:
                    await redis.delete(f"entity:{entity.id}:messages")

                # Message Generation
                try:
                    # Fetch nearby messages
                    messages = await fetch_nearby_messages(world, entities)

                    # Get the message generation prompt
                    message_prompt = prompts.get("message_generation_prompt", DEFAULT_MESSAGE_GENERATION_PROMPT)

                    # Construct and send the prompt to the LLM
                    message_result = await send_llm_request(
                        construct_prompt(message_prompt, world, messages)
                    )

                    # Process the result and save to Redis
                    if message_result.get("message"):
                        await redis.set(f"world:{world.world_id}:message", message_result["message"])
                        add_log(f"Message generated for World {world.world_id}: \"{message_result['message']}\".")
                except Exception as e:
                    logger.error(f"Error generating message for World {world.world_id}: {str(e)}")
                    add_log(f"Error generating message for World {world.world_id}: {str(e)}")

                # Throttle requests to the LLM
                await asyncio.sleep(REQUEST_DELAY)

                # Memory Generation
                try:
                    # Fetch nearby messages
                    messages = await fetch_nearby_messages(world, entities)

                    # Get the memory generation prompt
                    memory_prompt = prompts.get("memory_generation_prompt", DEFAULT_MEMORY_GENERATION_PROMPT)

                    # Construct and send the prompt to the LLM
                    memory_result = await send_llm_request(
                        construct_prompt(memory_prompt, world, messages)
                    )

                    # Process the result and save to Redis
                    if memory_result.get("memory"):
                        await redis.set(f"world:{world.world_id}:memory", memory_result["memory"])
                        add_log(f"Memory updated for World {world.world_id}: \"{memory_result['memory']}\".")
                except Exception as e:
                    logger.error(f"Error generating memory for World {world.world_id}: {str(e)}")
                    add_log(f"Error generating memory for World {world.world_id}: {str(e)}")

                # Throttle requests to the LLM
                await asyncio.sleep(REQUEST_DELAY)

                # Generate Movement (Per-Entity Processing)
                for entity in entities:
                    try:
                        # Construct the movement prompt for the current entity
                        movement_prompt = construct_prompt(
                            prompts.get("movement_generation_prompt", DEFAULT_MOVEMENT_GENERATION_PROMPT),
                            entity.dict(),
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
                        initial_position = (entity.x, entity.y)
                        if movement == "x+1":
                            entity.x = (entity.x + 1) % GRID_SIZE
                        elif movement == "x-1":
                            entity.x = (entity.x - 1) % GRID_SIZE
                        elif movement == "y+1":
                            entity.y = (entity.y + 1) % GRID_SIZE
                        elif movement == "y-1":
                            entity.y = (entity.y - 1) % GRID_SIZE
                        elif movement == "stay":
                            logger.info(f"Entity {entity.id} stays in place at {initial_position}.")
                            add_log(f"Entity {entity.id} stays in place at {initial_position}.")
                            continue
                        else:
                            logger.warning(f"Invalid movement command for Entity {entity.id}: {movement}")
                            add_log(f"Invalid movement command for Entity {entity.id}: {movement}")
                            continue

                        # Log and update position
                        logger.info(f"Entity {entity.id} moved from {initial_position} to ({entity.x}, {entity.y}) with action '{movement}'.")
                        add_log(f"Entity {entity.id} moved from {initial_position} to ({entity.x}, {entity.y}) with action '{movement}'.")

                        # Update the entity's position in Redis
                        await redis.hset(f"entity:{entity.id}", mapping={"x": entity.x, "y": entity.y})

                    except Exception as e:
                        logger.error(f"Error during movement generation for Entity {entity.id}: {str(e)}")
                        add_log(f"Error during movement generation for Entity {entity.id}: {str(e)}")

                    # Optional: Throttle requests to prevent overwhelming the LLM
                    await asyncio.sleep(REQUEST_DELAY)

            # Handle post-simulation steps
            logger.info(f"Completed {request.steps} step(s) for World {world_id}.")
            add_log(f"Simulation steps completed: {request.steps} step(s) for World {world_id}.")
            return JSONResponse({"status": f"Performed {request.steps} step(s) for World {world_id}."})

        except HTTPException as http_exc:
            # Re-raise HTTP exceptions to be handled by the custom exception handler
            raise http_exc
        except Exception as e:
            logger.error(f"Unexpected error during simulation steps: {str(e)}")
            add_log(f"Unexpected error during simulation steps: {str(e)}")
            raise HTTPException(status_code=500, detail=f"An error occurred during simulation steps: {str(e)}")

@app.post("/step", tags=["Simulation"])
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
                if isinstance(world, World):
                    try:
                        # Simulate world step
                        await simulate_world_step(world)

                        # Summarize state and submit to mTNN
                        summary_vector = world.summarize_state()
                        await submit_summary(world.world_id, summary_vector)

                        # Log the summary
                        add_log(f"World {world.world_id} summary: {summary_vector}")
                    except Exception as e:
                        logger.error(f"Error processing world {world.world_id}: {str(e)}")
                        add_log(f"Error processing world {world.world_id}: {str(e)}")
                else:
                    logger.error(f"Invalid object in app.state.worlds: {type(world)}")
                    add_log(f"Invalid object in app.state.worlds: {type(world)}")

            # Fetch entities per world
            for world in app.state.worlds:
                if isinstance(world, World):
                    entities = await get_entities_by_world(world.world_id)
                    logger.info(f"World {world.world_id}: Found {len(entities)} entities.")
                    add_log(f"World {world.world_id}: Found {len(entities)} entities.")

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
                    try:
                        # Fetch nearby messages
                        messages = await fetch_nearby_messages(world, entities)

                        # Get the message generation prompt
                        message_prompt = prompts.get("message_generation_prompt", DEFAULT_MESSAGE_GENERATION_PROMPT)

                        # Construct and send the prompt to the LLM
                        message_result = await send_llm_request(
                            construct_prompt(message_prompt, world, messages)
                        )

                        # Process the result and save to Redis
                        if "message" in message_result and message_result["message"]:
                            await redis.set(f"world:{world.world_id}:message", message_result["message"])
                            add_log(f"Message generated for World {world.world_id}: \"{message_result['message']}\".")
                    except Exception as e:
                        logger.error(f"Error generating message for World {world.world_id}: {str(e)}")
                        add_log(f"Error generating message for World {world.world_id}: {str(e)}")

                    # Throttle requests to the LLM
                    await asyncio.sleep(REQUEST_DELAY)

                    # Memory Generation
                    try:
                        # Fetch nearby messages
                        messages = await fetch_nearby_messages(world, entities)

                        # Get the memory generation prompt
                        memory_prompt = prompts.get("memory_generation_prompt", DEFAULT_MEMORY_GENERATION_PROMPT)

                        # Construct and send the prompt to the LLM
                        memory_result = await send_llm_request(
                            construct_prompt(memory_prompt, world, messages)
                        )

                        # Process the result and save to Redis
                        if "memory" in memory_result and memory_result["memory"]:
                            await redis.set(f"world:{world.world_id}:memory", memory_result["memory"])
                            add_log(f"Memory updated for World {world.world_id}: \"{memory_result['memory']}\".")
                    except Exception as e:
                        logger.error(f"Error generating memory for World {world.world_id}: {str(e)}")
                        add_log(f"Error generating memory for World {world.world_id}: {str(e)}")

                    # Throttle requests to the LLM
                    await asyncio.sleep(REQUEST_DELAY)

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

                            # Update the entity's position in Redis
                            await redis.hset(f"entity:{entity['id']}", mapping={"x": entity["x"], "y": entity["y"]})

                        except Exception as e:
                            logger.error(f"Error during movement generation for Entity {entity['id']}: {str(e)}")
                            add_log(f"Error during movement generation for Entity {entity['id']}: {str(e)}")

                        # Optional: Throttle requests to prevent overwhelming the LLM
                        await asyncio.sleep(REQUEST_DELAY)

            # Handle post-simulation steps
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
