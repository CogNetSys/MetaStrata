import os
import random
import asyncio
from typing import List
from fastapi import FastAPI, HTTPException, WebSocket, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.exceptions import HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel
from dotenv import load_dotenv
from redis.asyncio import Redis
from supabase import create_client, Client
import httpx
import logging
from asyncio import Semaphore
from contextlib import asynccontextmanager
from typing import List
import traceback
from collections import deque
from datetime import datetime

from config import (
    SUPABASE_URL,
    SUPABASE_KEY,
    GROQ_API_KEY,
    GROQ_API_ENDPOINT,
    REDIS_ENDPOINT,
    REDIS_PASSWORD,
    GRID_SIZE,
    NUM_AGENTS,
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

# Global log queue for real-time logging
LOG_QUEUE = deque(maxlen=200)  # Keeps the last 200 log messages

# Define the configuration model
class SimulationSettings(BaseModel):
    grid_size: int = GRID_SIZE
    num_agents: int = NUM_AGENTS
    max_steps: int = MAX_STEPS
    chebyshev_distance: int = CHEBYSHEV_DISTANCE
    llm_model: str = LLM_MODEL
    llm_max_tokens: int = LLM_MAX_TOKENS
    llm_temperature: float = LLM_TEMPERATURE
    request_delay: float = REQUEST_DELAY
    max_concurrent_requests: int = MAX_CONCURRENT_REQUESTS

# Simulation Settings Model
class SimulationSettings(BaseModel):
    grid_size: int = 30
    num_agents: int = 10
    max_steps: int = 100
    chebyshev_distance: int = 5
    llm_model: str = "llama-3.1-8b-instant"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.7
    request_delay: float = 2.2
    max_concurrent_requests: int = 1

# Prompt Settings Model
class PromptSettings(BaseModel):
    message_generation_prompt: str
    memory_generation_prompt: str
    movement_generation_prompt: str

# Agent Model (Pydantic)
class Agent(BaseModel):
    id: int
    name: str
    x: int
    y: int
    memory: str = ""  # Default empty memory for new agents

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

# Fetching Neraby Agents Function
def chebyshev_distance(x1, y1, x2, y2):
    return max(abs(x1 - x2), abs(y1 - y2))

# Data Models
class StepRequest(BaseModel):
    steps: int

# Helper Functions
def chebyshev_distance(x1, y1, x2, y2):
    dx = min(abs(x1 - x2), GRID_SIZE - abs(x1 - x2))
    dy = min(abs(y1 - y2), GRID_SIZE - abs(y1 - y2))
    return max(dx, dy)

def construct_prompt(template, agent, messages):
    messages_str = "\n".join(messages) if messages else "No recent messages."
    memory = agent.get("memory", "No prior memory.")
    return template.format(
        agentId=agent["id"], x=agent["x"], y=agent["y"],
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
        
# Chebyshev Distance Helper Function for calculating the distance for Nearby Agents function.
def chebyshev_distance(x1, y1, x2, y2):
    return max(abs(x1 - x2), abs(y1 - y2))
        
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
                    elif "create a summary" in prompt:
                        await asyncio.sleep(REQUEST_DELAY)
                        return {"memory": content}
                    elif "decide your next move" in prompt:
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

async def initialize_agents():
    logger.info("Resetting simulation state.")
    await redis.flushdb()  # Clear all Redis data, including agent_keys

    agents = [
        {
            "id": i,
            "name": f"Agent-{i}",
            "x": random.randint(0, GRID_SIZE - 1),
            "y": random.randint(0, GRID_SIZE - 1),
            "memory": ""
        }
        for i in range(NUM_AGENTS)
    ]

    for agent in agents:
        agent_key = f"agent:{agent['id']}"
        await redis.hset(agent_key, mapping=agent)
        await redis.lpush("agent_keys", agent_key)  # Add to agent_keys list
        await redis.delete(f"{agent['id']}:messages")  # Clear message queue

    logger.info("Agents initialized.")
    return agents

async def fetch_nearby_messages(agent, agents, message_to_send=None):
    """
    Fetch messages from nearby agents or optionally send a message to them.
    Args:
        agent (dict): The current agent's data.
        agents (list): List of all agents.
        message_to_send (str): Optional message to send to nearby agents.
    Returns:
        list: A list of messages received from nearby agents.
    """
    nearby_agents = [
        a for a in agents if a["id"] != agent["id"] and chebyshev_distance(agent["x"], agent["y"], a["x"], a["y"]) <= CHEBYSHEV_DISTANCE
    ]
    received_messages = []

    for nearby_agent in nearby_agents:
        # Fetch existing messages from the nearby agent
        msg = await redis.hget(f"agent:{nearby_agent['id']}", "message")
        logger.info(f"Fetched message for agent {nearby_agent['id']}: {msg}")
        if msg:
            received_messages.append(msg)

        # If a message is being sent, add it to the recipient's queue
        if message_to_send:
            recipient_key = f"agent:{nearby_agent['id']}:messages"
            await redis.lpush(recipient_key, f"From Agent {agent['id']}: {message_to_send}")
            logger.info(f"Sent message from Agent {agent['id']} to Agent {nearby_agent['id']}")

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
    lifespan=lifespan)

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

# Simulation API Endpoints
@app.post("/reset", tags=["World Simulation"])
async def reset_simulation():
    global stop_signal
    try:
        # Log reset initiation
        add_log("Reset simulation process initiated.")
        
        stop_signal = False  # Reset stop signal before starting
        add_log("Stop signal reset to False.")
        
        # Clear Redis database
        await redis.flushdb()
        add_log("Redis database flushed successfully.")
        
        # Initialize agents
        agents = await initialize_agents()
        add_log(f"Agents reinitialized successfully. Total agents: {len(agents)}")
        
        # Log successful reset
        add_log("Simulation reset completed successfully.")
        return JSONResponse({"status": "Simulation reset successfully.", "agents": agents})
    except Exception as e:
        # Log any error encountered during reset
        error_message = f"Error during simulation reset: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.post("/initialize", tags=["World Simulation"])
async def initialize_simulation():
    global stop_signal
    try:
        # Log the initiation of the simulation
        add_log("Simulation initialization process started.")
        
        # Reset the stop signal
        stop_signal = False
        add_log("Stop signal reset to False.")
        
        # Initialize agents
        agents = await initialize_agents()
        add_log(f"Agents initialized successfully. Total agents: {len(agents)}")
        
        # Log success and return the response
        add_log("Simulation started successfully.")
        return JSONResponse({"status": "Simulation started successfully.", "agents": agents})
    except Exception as e:
        # Log and raise an error if the initialization fails
        error_message = f"Error during simulation initialization: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

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
                logger.info("Stopping steps due to stop signal.")
                add_log("Simulation steps halted by stop signal.")
                break

            # Fetch all agent keys dynamically from Redis
            agent_keys = await redis.keys("agent:*")  # Match all agent keys
            if not agent_keys:
                logger.warning("No agents found in Redis!")
                add_log("No agents found in Redis. Aborting simulation steps.")
                return JSONResponse({"status": "No agents to process."})

            logger.info(f"Step {step + 1}: Found {len(agent_keys)} agents.")
            add_log(f"Step {step + 1}: Found {len(agent_keys)} agents.")

            # Filter keys to ensure only valid hashes are processed
            valid_agent_keys = []
            for key in agent_keys:
                key_type = await redis.type(key)
                if key_type == "hash":
                    valid_agent_keys.append(key)
                else:
                    logger.warning(f"Skipping invalid key {key} of type {key_type}")
                    add_log(f"Skipping invalid key {key} of type {key_type}")

            # Fetch agent data from Redis for all valid keys
            agents = [
                {
                    "id": int(agent_data["id"]),
                    "name": agent_data["name"],
                    "x": int(agent_data["x"]),
                    "y": int(agent_data["y"]),
                    "memory": agent_data.get("memory", "")
                }
                for agent_data in await asyncio.gather(*[redis.hgetall(key) for key in valid_agent_keys])
                if agent_data  # Ensure we only include valid agent data
            ]

            logger.info(f"Processing {len(agents)} agents.")
            add_log(f"Processing {len(agents)} agents for step {step + 1}.")

            # Process incoming messages for each agent
            for agent in agents:
                try:
                    # Fetch the existing message field
                    message = await redis.hget(f"agent:{agent['id']}", "message")

                    if message:
                        logger.info(f"Agent {agent['id']} received message: {message}")
                        add_log(f"Agent {agent['id']} received message: {message}")

                        # Optionally update memory or trigger actions based on the message
                        updated_memory = f"{agent['memory']} | Received: {message}"
                        await redis.hset(f"agent:{agent['id']}", "memory", updated_memory)

                        # Clear the message field after processing (if required)
                        await redis.hset(f"agent:{agent['id']}", "message", "")
                except Exception as e:
                    logger.error(f"Error processing message for Agent {agent['id']}: {str(e)}")
                    add_log(f"Error processing message for Agent {agent['id']}: {str(e)}")

            # Clear message queues only after processing all agents
            for agent in agents:
                await redis.delete(f"agent:{agent['id']}:messages")

            # Message Generation
            for agent in agents:
                try:
                    messages = await fetch_nearby_messages(agent, agents)
                    message_prompt = prompts.get("message_generation_prompt", DEFAULT_MESSAGE_GENERATION_PROMPT)
                    message_result = await send_llm_request(
                        construct_prompt(message_prompt, agent, messages)
                    )
                    if "message" in message_result:
                        await redis.hset(f"agent:{agent['id']}", "message", message_result["message"])
                        add_log(f"Message generated for Agent {agent['id']}: {message_result['message']}")
                except Exception as e:
                    logger.error(f"Error generating message for Agent {agent['id']}: {str(e)}")
                    add_log(f"Error generating message for Agent {agent['id']}: {str(e)}")
                await asyncio.sleep(REQUEST_DELAY)

            if stop_signal:
                logger.info("Stopping after message generation due to stop signal.")
                add_log("Simulation steps halted after message generation by stop signal.")
                break

            # Memory Generation
            for agent in agents:
                try:
                    messages = await fetch_nearby_messages(agent, agents)
                    memory_prompt = prompts.get("memory_generation_prompt", DEFAULT_MEMORY_GENERATION_PROMPT)
                    memory_result = await send_llm_request(
                        construct_prompt(memory_prompt, agent, messages)
                    )
                    if "memory" in memory_result:
                        await redis.hset(f"agent:{agent['id']}", "memory", memory_result["memory"])
                        add_log(f"Memory updated for Agent {agent['id']}: {memory_result['memory']}")
                except Exception as e:
                    logger.error(f"Error generating memory for Agent {agent['id']}: {str(e)}")
                    add_log(f"Error generating memory for Agent {agent['id']}: {str(e)}")
                await asyncio.sleep(REQUEST_DELAY)

            if stop_signal:
                logger.info("Stopping after memory generation due to stop signal.")
                add_log("Simulation steps halted after memory generation by stop signal.")
                break

            # Movement Generation
            for agent in agents:
                try:
                    movement_prompt = prompts.get("movement_generation_prompt", DEFAULT_MOVEMENT_GENERATION_PROMPT)
                    movement_result = await send_llm_request(
                        construct_prompt(movement_prompt, agent, [])
                    )
                    if "movement" in movement_result:
                        movement = movement_result["movement"].strip().lower()
                        initial_position = (agent["x"], agent["y"])

                        # Apply movement logic
                        if movement == "x+1":
                            agent["x"] = (agent["x"] + 1) % GRID_SIZE
                        elif movement == "x-1":
                            agent["x"] = (agent["x"] - 1) % GRID_SIZE
                        elif movement == "y+1":
                            agent["y"] = (agent["y"] + 1) % GRID_SIZE
                        elif movement == "y-1":
                            agent["y"] = (agent["y"] - 1) % GRID_SIZE
                        elif movement == "stay":
                            logger.info(f"Agent {agent['id']} stays in place at {initial_position}.")
                            add_log(f"Agent {agent['id']} stays in place at {initial_position}.")
                            continue
                        else:
                            logger.warning(f"Invalid movement command for Agent {agent['id']}: {movement}")
                            add_log(f"Invalid movement command for Agent {agent['id']}: {movement}")
                            continue

                        # Log and update position
                        logger.info(f"Agent {agent['id']} moved from {initial_position} to ({agent['x']}, {agent['y']}) with action '{movement}'.")
                        add_log(f"Agent {agent['id']} moved from {initial_position} to ({agent['x']}, {agent['y']}) with action '{movement}'.")
                        await redis.hset(f"agent:{agent['id']}", mapping={"x": agent["x"], "y": agent["y"]})
                except Exception as e:
                    logger.error(f"Error generating movement for Agent {agent['id']}: {str(e)}")
                    add_log(f"Error generating movement for Agent {agent['id']}: {str(e)}")
                await asyncio.sleep(REQUEST_DELAY)

        logger.info(f"Completed {request.steps} step(s).")
        add_log(f"Simulation steps completed: {request.steps} step(s).")
        return JSONResponse({"status": f"Performed {request.steps} step(s)."})
    except Exception as e:
        logger.error(f"Unexpected error during simulation steps: {str(e)}")
        add_log(f"Unexpected error during simulation steps: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during simulation steps: {str(e)}")

@app.post("/stop", tags=["World Simulation"])
async def stop_simulation():
    global stop_signal
    try:
        # Log the start of the stop process
        add_log("Stop simulation process initiated.")
        
        # Set the stop signal
        stop_signal = True
        add_log("Stop signal triggered successfully.")
        
        # Log successful completion
        add_log("Simulation stopping process completed.")
        return JSONResponse({"status": "Simulation stopping."})
    except Exception as e:
        # Log and raise any unexpected errors
        error_message = f"Error during simulation stop process: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/agents", response_model=Agent, tags=["Agents"])
async def create_agent(agent: Agent):
    agent_key = f"agent:{agent.id}"

    try:
        # Log the attempt to create a new agent
        add_log(f"Attempting to create new agent with ID {agent.id} and name '{agent.name}'.")

        # Check if the ID already exists in Redis
        if await redis.exists(agent_key):
            error_message = f"Agent ID {agent.id} already exists in Redis."
            add_log(error_message)
            raise HTTPException(status_code=400, detail=error_message)

        # Check if the ID already exists in Supabase
        existing_agent = supabase.table("agents").select("id").eq("id", agent.id).execute()
        if existing_agent.data:
            error_message = f"Agent ID {agent.id} already exists in Supabase."
            add_log(error_message)
            raise HTTPException(status_code=400, detail=error_message)

        # Save agent data in Redis
        await redis.hset(agent_key, mapping=agent.dict())
        add_log(f"Agent data for ID {agent.id} saved in Redis.")

        # Add the agent key to the Redis list
        await redis.lpush("agent_keys", agent_key)
        add_log(f"Agent key for ID {agent.id} added to Redis agent_keys list.")

        # Save the agent in Supabase
        supabase.table("agents").insert(agent.dict()).execute()
        add_log(f"Agent data for ID {agent.id} saved in Supabase.")

        # Log the successful creation of the agent
        add_log(f"New agent created successfully: ID={agent.id}, Name={agent.name}, Position=({agent.x}, {agent.y})")
        
        return agent

    except Exception as e:
        # Log and raise unexpected errors
        error_message = f"Error creating agent with ID {agent.id}: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)


@app.get("/agents/{agent_id}", response_model=Agent, tags=["Agents"])
async def get_agent(agent_id: int):
    # Log the attempt to fetch agent data
    add_log(f"Fetching data for agent with ID {agent_id}.")

    # Fetch agent data from Redis
    agent_data = await redis.hgetall(f"agent:{agent_id}")
    if not agent_data:
        error_message = f"Agent with ID {agent_id} not found."
        add_log(error_message)
        raise HTTPException(status_code=404, detail=error_message)

    # Convert Redis data into an Agent model (ensure proper types are cast)
    try:
        agent = Agent(
            id=agent_id,
            name=agent_data.get("name"),
            x=int(agent_data.get("x")),
            y=int(agent_data.get("y")),
            memory=agent_data.get("memory", "")
        )
        add_log(f"Successfully fetched agent with ID {agent_id}, Name: {agent.name}, Position: ({agent.x}, {agent.y}).")
        return agent
    except Exception as e:
        error_message = f"Failed to parse data for agent ID {agent_id}: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.put("/agents/{agent_id}", response_model=Agent, tags=["Agents"])
async def update_agent(agent_id: int, agent: Agent):
    try:
        # Log the attempt to update agent data
        add_log(f"Attempting to update agent with ID {agent_id}.")

        # Update agent data in Redis
        await redis.hset(f"agent:{agent_id}", mapping=agent.dict())
        add_log(f"Agent with ID {agent_id} updated in Redis.")

        # Update agent in Supabase
        supabase.table("agents").update(agent.dict()).eq("id", agent_id).execute()
        add_log(f"Agent with ID {agent_id} updated in Supabase.")

        # Log successful update
        add_log(f"Successfully updated agent with ID {agent_id}.")
        return agent
    except Exception as e:
        # Log and raise error if update fails
        error_message = f"Error updating agent with ID {agent_id}: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.delete("/agents/{agent_id}", tags=["Agents"])
async def delete_agent(agent_id: int):
    agent_key = f"agent:{agent_id}"
    try:
        # Log the attempt to delete an agent
        add_log(f"Attempting to delete agent with ID {agent_id}.")

        # Delete agent from Redis
        await redis.delete(agent_key)
        add_log(f"Agent with ID {agent_id} deleted from Redis.")

        # Remove the key from the Redis list
        await redis.lrem("agent_keys", 0, agent_key)
        add_log(f"Agent key with ID {agent_id} removed from Redis agent_keys list.")

        # Optionally, delete the agent from Supabase
        supabase.table("agents").delete().eq("id", agent_id).execute()
        add_log(f"Agent with ID {agent_id} deleted from Supabase.")

        return {"status": "Agent deleted successfully"}
    except Exception as e:
        # Log and raise any unexpected errors
        error_message = f"Error deleting agent with ID {agent_id}: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

from fastapi import Query

@app.post("/agents/{recipient_id}/create_memory", tags=["Utilities"])
async def create_memory(
    recipient_id: int,
    message: str = Query(..., description="The memory content to add or update for the recipient agent.")
):
    recipient_key = f"agent:{recipient_id}"

    try:
        # Log the attempt to create memory
        add_log(f"Creating a memory for Agent {recipient_id}: \"{message}\".")

        # Validate that the recipient exists
        if not await redis.exists(recipient_key):
            error_message = f"Recipient Agent ID {recipient_id} not found."
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
        add_log(f"Memory updated successfully for Agent {recipient_id}: \"{message}\".")

        return {"status": "Memory updated successfully", "message": message}

    except Exception as e:
        # Log and raise any unexpected errors
        error_message = f"Error creating memory for Agent {recipient_id}: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/agents/{agent_id}/nearby", response_model=List[Agent], tags=["Utilities"])
async def get_nearby_agents(agent_id: int):
    try:
        # Log the attempt to fetch nearby agents
        add_log(f"Fetching nearby agents for Agent ID {agent_id}.")

        # Get the agent's position from Redis
        agent_data = await redis.hgetall(f"agent:{agent_id}")
        if not agent_data:
            error_message = f"Agent with ID {agent_id} not found."
            add_log(error_message)
            raise HTTPException(status_code=404, detail=error_message)

        agent = Agent(**agent_data)

        # Fetch all agents except the current one
        all_agents = [
            Agent(**await redis.hgetall(f"agent:{i}"))
            for i in range(NUM_AGENTS) if i != agent_id
        ]

        # Filter nearby agents based on Chebyshev distance
        nearby_agents = [
            a for a in all_agents
            if chebyshev_distance(agent.x, agent.y, a.x, a.y) <= CHEBYSHEV_DISTANCE
        ]

        add_log(f"Nearby agents fetched successfully for Agent ID {agent_id}. Total nearby agents: {len(nearby_agents)}.")
        return nearby_agents
    except Exception as e:
        # Log and raise any unexpected errors
        error_message = f"Error fetching nearby agents for Agent ID {agent_id}: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.post("/sync_agents", tags=["Utilities"])
async def sync_agents():
    try:
        # Log the start of the synchronization process
        add_log("Synchronization process initiated between Redis and Supabase.")

        all_agents = [
            Agent(**await redis.hgetall(f"agent:{i}"))
            for i in range(NUM_AGENTS)
        ]

        for agent in all_agents:
            supabase.table("agents").upsert(agent.dict()).execute()
            add_log(f"Agent with ID {agent.id} synchronized to Supabase.")

        add_log("Synchronization process completed successfully.")
        return {"status": "Agents synchronized between Redis and Supabase"}
    except Exception as e:
        # Log and raise any unexpected errors
        error_message = f"Error during agent synchronization: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/settings", response_model=SimulationSettings, tags=["Settings"])
async def get_settings():
    try:
        # Log the attempt to fetch simulation settings
        add_log("Fetching simulation settings.")

        # Fetch all agent keys from Redis
        agent_keys = await redis.keys("agent:*")  # Get all keys matching agent pattern
        num_agents = len(agent_keys)  # Count the number of agents

        # Log successful retrieval
        add_log(f"Simulation settings fetched successfully. Total agents: {num_agents}.")
        
        # Return the dynamically updated number of agents
        return SimulationSettings(
            grid_size=GRID_SIZE,
            num_agents=num_agents,
            max_steps=MAX_STEPS,
            chebyshev_distance=CHEBYSHEV_DISTANCE,
            llm_model=LLM_MODEL,
            llm_max_tokens=LLM_MAX_TOKENS,
            llm_temperature=LLM_TEMPERATURE,
            request_delay=REQUEST_DELAY,
            max_concurrent_requests=MAX_CONCURRENT_REQUESTS
        )
    except Exception as e:
        # Log and raise any unexpected errors
        error_message = f"Error fetching simulation settings: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.post("/settings", response_model=SimulationSettings, tags=["Settings"])
async def set_settings(settings: SimulationSettings):
    try:
        # Log the attempt to update simulation settings
        add_log("Updating simulation settings.")

        global GRID_SIZE, NUM_AGENTS, MAX_STEPS, CHEBYSHEV_DISTANCE, LLM_MODEL
        global LLM_MAX_TOKENS, LLM_TEMPERATURE, REQUEST_DELAY, MAX_CONCURRENT_REQUESTS

        GRID_SIZE = settings.grid_size
        NUM_AGENTS = settings.num_agents
        MAX_STEPS = settings.max_steps
        CHEBYSHEV_DISTANCE = settings.chebyshev_distance
        LLM_MODEL = settings.llm_model
        LLM_MAX_TOKENS = settings.llm_max_tokens
        LLM_TEMPERATURE = settings.llm_temperature
        REQUEST_DELAY = settings.request_delay
        MAX_CONCURRENT_REQUESTS = settings.max_concurrent_requests

        # Log successful update
        add_log("Simulation settings updated successfully.")
        return settings
    except Exception as e:
        # Log and raise any unexpected errors
        error_message = f"Error updating simulation settings: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

# Endpoint to get current prompt templates
@app.get("/prompts", response_model=PromptSettings, tags=["Settings"])
async def get_prompts():
    try:
        # Log the attempt to fetch prompt templates
        add_log("Fetching current prompt templates.")

        # Return the prompt templates
        prompts = PromptSettings(
            message_generation_prompt=DEFAULT_MESSAGE_GENERATION_PROMPT,
            memory_generation_prompt=DEFAULT_MEMORY_GENERATION_PROMPT,
            movement_generation_prompt=DEFAULT_MOVEMENT_GENERATION_PROMPT
        )

        # Log successful retrieval
        add_log("Prompt templates fetched successfully.")
        return prompts
    except Exception as e:
        # Log and raise any unexpected errors
        error_message = f"Error fetching prompt templates: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

# Endpoint to set new prompt templates
@app.post("/prompts", response_model=PromptSettings, tags=["Settings"])
async def set_prompts(prompts: PromptSettings):
    try:
        # Log the attempt to update prompt templates
        add_log("Updating prompt templates.")

        global DEFAULT_MESSAGE_GENERATION_PROMPT, DEFAULT_MEMORY_GENERATION_PROMPT, DEFAULT_MOVEMENT_GENERATION_PROMPT

        DEFAULT_MESSAGE_GENERATION_PROMPT = prompts.message_generation_prompt
        DEFAULT_MEMORY_GENERATION_PROMPT = prompts.memory_generation_prompt
        DEFAULT_MOVEMENT_GENERATION_PROMPT = prompts.movement_generation_prompt

        # Log successful update
        add_log("Prompt templates updated successfully.")
        return prompts
    except Exception as e:
        # Log and raise any unexpected errors
        error_message = f"Error updating prompt templates: {str(e)}"
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
