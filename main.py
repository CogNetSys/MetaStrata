import os
import random
import asyncio
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from redis.asyncio import Redis
from supabase import create_client, Client
import httpx
import logging
from asyncio import Semaphore
from contextlib import asynccontextmanager
from typing import List

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


# Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
REDIS_ENDPOINT = "cute-crawdad-25113.upstash.io"
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# Simulation Configuration (Initial Defaults)
GRID_SIZE = 30  # 30x30 grid
NUM_AGENTS = 10
MAX_STEPS = 100
CHEBYSHEV_DISTANCE = 5
LLM_MODEL = "llama-3.2-11b-vision-preview"
LLM_MAX_TOKENS = 2048
LLM_TEMPERATURE = 0.7
REQUEST_DELAY = 2.2  # Fixed delay in seconds between requests
MAX_CONCURRENT_REQUESTS = 5  # Limit concurrent requests to prevent rate limiting

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

# Prompt Templates
GRID_DESCRIPTION = "The field size is 30 x 30 with periodic boundary conditions, and there are a total of 10 beings. You are free to move around the field and converse with other beings."

DEFAULT_MESSAGE_GENERATION_PROMPT = """
[INST]
You are being{agentId} at position ({x}, {y}). {grid_description} You have a summary memory of the situation so far: {memory}. You received messages from the surrounding beings: {messages}. Based on the above, you send a message to the surrounding beings. Your message will reach beings up to distance {distance} away. What message do you send?
Respond with only the message content, and nothing else.
[/INST]
"""

DEFAULT_MEMORY_GENERATION_PROMPT = """
[INST]
You are being{agentId} at position ({x}, {y}). {grid_description} You have a summary memory of the situation so far: {memory}. You received messages from the surrounding beings: {messages}. Based on the above, summarize the situation you and the other beings have been in so far for you to remember.
Respond with only the summary, and nothing else.
[/INST]
"""

DEFAULT_MOVEMENT_GENERATION_PROMPT = """
[INST]
You are being{agentId} at position ({x}, {y}). {grid_description} You have a summary memory of the situation so far: {memory}.
Based on the above, choose your next move. Respond with only one of the following options, and nothing else: "x+1", "x-1", "y+1", "y-1", or "stay".
Do not provide any explanation or additional text.
[/INST]
"""

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
app = FastAPI(lifespan=lifespan)

# Simulation API Endpoints
@app.post("/reset")
async def reset_simulation():
    global stop_signal
    stop_signal = False  # Reset stop signal before starting
    await redis.flushdb()
    agents = await initialize_agents()
    return JSONResponse({"status": "Simulation reset successfully.", "agents": agents})

@app.post("/start")
async def start_simulation():
    global stop_signal
    stop_signal = False  # Reset stop signal before starting
    agents = await initialize_agents()
    return JSONResponse({"status": "Simulation started successfully.", "agents": agents})

@app.post("/step")
async def perform_steps(request: StepRequest):
    global stop_signal
    stop_signal = False  # Reset stop signal before starting steps

    # Fetch the current prompt templates from FastAPI
    prompts = await fetch_prompts_from_fastapi()

    logger.info("Starting simulation steps...")

    for step in range(request.steps):
        if stop_signal:
            logger.info("Stopping steps due to stop signal.")
            break

        # Fetch all agent keys dynamically from Redis
        agent_keys = await redis.keys("agent:*")  # Match all agent keys
        if not agent_keys:
            logger.warning("No agents found in Redis!")
            return JSONResponse({"status": "No agents to process."})

        logger.info(f"Step {step + 1}: Found {len(agent_keys)} agents.")

        # Filter keys to ensure only valid hashes are processed
        valid_agent_keys = []
        for key in agent_keys:
            key_type = await redis.type(key)
            if key_type == "hash":
                valid_agent_keys.append(key)
            else:
                logger.warning(f"Skipping invalid key {key} of type {key_type}")

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

        # Process incoming messages for each agent
        for agent in agents:
            # Fetch the existing message field
            message = await redis.hget(f"agent:{agent['id']}", "message")
            
            if message:
                logger.info(f"Agent {agent['id']} received message: {message}")

                # Optionally update memory or trigger actions based on the message
                updated_memory = f"{agent['memory']} | Received: {message}"
                await redis.hset(f"agent:{agent['id']}", "memory", updated_memory)

                # Clear the message field after processing (if required)
                await redis.hset(f"agent:{agent['id']}", "message", "")

        # Clear message queues only after processing all agents
        for agent in agents:
            await redis.delete(f"agent:{agent['id']}:messages")


        # Message Generation
        for agent in agents:
            messages = await fetch_nearby_messages(agent, agents)
            message_prompt = prompts.get("message_generation_prompt", DEFAULT_MESSAGE_GENERATION_PROMPT)
            message_result = await send_llm_request(
                construct_prompt(message_prompt, agent, messages)
            )
            if "message" in message_result:
                await redis.hset(f"agent:{agent['id']}", "message", message_result["message"])
            await asyncio.sleep(REQUEST_DELAY)

        if stop_signal:
            logger.info("Stopping after message generation due to stop signal.")
            break

        # Memory Generation
        for agent in agents:
            messages = await fetch_nearby_messages(agent, agents)
            memory_prompt = prompts.get("memory_generation_prompt", DEFAULT_MEMORY_GENERATION_PROMPT)
            memory_result = await send_llm_request(
                construct_prompt(memory_prompt, agent, messages)
            )
            if "memory" in memory_result:
                await redis.hset(f"agent:{agent['id']}", "memory", memory_result["memory"])
            await asyncio.sleep(REQUEST_DELAY)

        if stop_signal:
            logger.info("Stopping after memory generation due to stop signal.")
            break

        # Movement Generation
        for agent in agents:
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
                    continue
                else:
                    logger.warning(f"Invalid movement command for Agent {agent['id']}: {movement}")
                    continue

                # Log and update position
                logger.info(f"Agent {agent['id']} moved from {initial_position} to ({agent['x']}, {agent['y']}) with action '{movement}'.")
                await redis.hset(f"agent:{agent['id']}", mapping={"x": agent["x"], "y": agent["y"]})
            await asyncio.sleep(REQUEST_DELAY)

    logger.info(f"Completed {request.steps} step(s).")
    return JSONResponse({"status": f"Performed {request.steps} step(s)."})

@app.post("/stop")
async def stop_simulation():
    global stop_signal
    stop_signal = True
    logger.info("Stop signal triggered.")
    return JSONResponse({"status": "Simulation stopping."})

@app.post("/agents", response_model=Agent)
async def create_agent(agent: Agent):
    agent_key = f"agent:{agent.id}"

    # Check if the ID already exists in Redis
    if await redis.exists(agent_key):
        raise HTTPException(status_code=400, detail=f"Agent ID {agent.id} already exists in Redis.")

    # Check if the ID already exists in Supabase
    existing_agent = supabase.table("agents").select("id").eq("id", agent.id).execute()
    if existing_agent.data:
        raise HTTPException(status_code=400, detail=f"Agent ID {agent.id} already exists in Supabase.")

    # Save agent data in Redis
    await redis.hset(agent_key, mapping=agent.dict())

    # Add the agent key to the Redis list
    await redis.lpush("agent_keys", agent_key)

    # Save the agent in Supabase
    supabase.table("agents").insert(agent.dict()).execute()

    return agent

@app.get("/agents/{agent_id}", response_model=Agent)
async def get_agent(agent_id: int):
    # Fetch agent data from Redis
    agent_data = await redis.hgetall(f"agent:{agent_id}")
    if not agent_data:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Convert Redis data into an Agent model (ensure proper types are cast)
    agent = Agent(
        id=agent_id,
        name=agent_data.get("name"),
        x=int(agent_data.get("x")),
        y=int(agent_data.get("y")),
        memory=agent_data.get("memory", "")
    )
    return agent

@app.put("/agents/{agent_id}", response_model=Agent)
async def update_agent(agent_id: int, agent: Agent):
    # Update agent data in Redis (assuming 'hset' is used to update fields)
    await redis.hset(f"agent:{agent_id}", mapping=agent.dict())
    
    # Optionally, update agent in Supabase for persistent storage
    supabase.table("agents").update(agent.dict()).eq("id", agent_id).execute()
    
    return agent

@app.post("/agents/{agent_id}/send_memory")
async def send_memory(agent_id: int, recipient_id: int, message: str):
    sender_key = f"agent:{agent_id}"
    recipient_key = f"agent:{recipient_id}"

    # Validate that the sender and recipient exist
    if not await redis.exists(sender_key):
        raise HTTPException(status_code=404, detail=f"Sender Agent ID {agent_id} not found.")
    if not await redis.exists(recipient_key):
        raise HTTPException(status_code=404, detail=f"Recipient Agent ID {recipient_id} not found.")

    # Fetch the recipient's existing memory field
    existing_memory = await redis.hget(recipient_key, "memory")
    if existing_memory:
        # Append the new message to the recipient's memory
        updated_memory = f"{existing_memory}\nFrom Agent {agent_id}: {message}"
    else:
        # Start the memory with the new message
        updated_memory = f"From Agent {agent_id}: {message}"

    # Update the recipient's memory field
    await redis.hset(recipient_key, "memory", updated_memory)

    logger.info(f'Message appended to Agent {recipient_id}\'s memory from Agent {agent_id}: "{message}"')
    return {"status": "Memory updated successfully", "message": message}

@app.delete("/agents/{agent_id}")
async def delete_agent(agent_id: int):
    agent_key = f"agent:{agent_id}"

    # Delete agent from Redis
    await redis.delete(agent_key)

    # Remove the key from the Redis list
    await redis.lrem("agent_keys", 0, agent_key)

    # Optionally, delete the agent from Supabase
    supabase.table("agents").delete().eq("id", agent_id).execute()

    return {"status": "Agent deleted successfully"}

@app.get("/agents/{agent_id}/nearby", response_model=List[Agent])
async def get_nearby_agents(agent_id: int):
    # Get the agent's position from Redis
    agent_data = await redis.hgetall(f"agent:{agent_id}")
    if not agent_data:
        raise HTTPException(status_code=404, detail="Agent not found")
    
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
    
    return nearby_agents

@app.post("/sync_agents")
async def sync_agents():
    all_agents = [
        Agent(**await redis.hgetall(f"agent:{i}"))
        for i in range(NUM_AGENTS)
    ]
    
    for agent in all_agents:
        supabase.table("agents").upsert(agent.dict()).execute()
    
    return {"status": "Agents synchronized between Redis and Supabase"}

@app.get("/settings", response_model=SimulationSettings)
async def get_settings():
    # Fetch all agent keys from Redis
    agent_keys = await redis.keys("agent:*")  # Get all keys matching agent pattern
    num_agents = len(agent_keys)  # Count the number of agents

    # Alternatively, you can count agents from Supabase
    # num_agents = len(supabase.table("agents").select("id").execute().data)

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

@app.post("/settings", response_model=SimulationSettings)
async def set_settings(settings: SimulationSettings):
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

    return settings

# Endpoint to get current prompt templates
@app.get("/prompts", response_model=PromptSettings)
async def get_prompts():
    return PromptSettings(
        message_generation_prompt=DEFAULT_MESSAGE_GENERATION_PROMPT,
        memory_generation_prompt=DEFAULT_MEMORY_GENERATION_PROMPT,
        movement_generation_prompt=DEFAULT_MOVEMENT_GENERATION_PROMPT
    )

# Endpoint to set new prompt templates
@app.post("/prompts", response_model=PromptSettings)
async def set_prompts(prompts: PromptSettings):
    global DEFAULT_MESSAGE_GENERATION_PROMPT, DEFAULT_MEMORY_GENERATION_PROMPT, DEFAULT_MOVEMENT_GENERATION_PROMPT

    DEFAULT_MESSAGE_GENERATION_PROMPT = prompts.message_generation_prompt
    DEFAULT_MEMORY_GENERATION_PROMPT = prompts.memory_generation_prompt
    DEFAULT_MOVEMENT_GENERATION_PROMPT = prompts.movement_generation_prompt

    return prompts

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)