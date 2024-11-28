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

load_dotenv()

# Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
REDIS_ENDPOINT = "cute-crawdad-25113.upstash.io"
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# Simulation Configuration
GRID_SIZE = 30  # 30x30 grid
NUM_AGENTS = 10
MAX_STEPS = 100
CHEBYSHEV_DISTANCE = 5
LLM_MODEL = "llama-3.2-11b-vision-preview"
LLM_MAX_TOKENS = 2048
LLM_TEMPERATURE = 0.7
REQUEST_DELAY = 2.5  # Fixed delay in seconds between requests
MAX_CONCURRENT_REQUESTS = 1  # Limit concurrent requests to prevent rate limiting

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

# Prompt Templates
GRID_DESCRIPTION = "The field size is 30 x 30 with periodic boundary conditions, and there are a total of 10 beings. You are free to move around the field and converse with other beings."

MESSAGE_GENERATION_PROMPT = """
[INST]
You are being{agentId} at position ({x}, {y}). {grid_description} You have a summary memory of the situation so far: {memory}. You received messages from the surrounding beings: {messages}. Based on the above, you send a message to the surrounding beings. Your message will reach beings up to distance {distance} away. What message do you send?
Respond with only the message content, and nothing else.
[/INST]
"""

MEMORY_GENERATION_PROMPT = """
[INST]
You are being{agentId} at position ({x}, {y}). {grid_description} You have a summary memory of the situation so far: {memory}. You received messages from the surrounding beings: {messages}. Based on the above, summarize the situation you and the other beings have been in so far for you to remember.
Respond with only the summary, and nothing else.
[/INST]
"""

MOVEMENT_GENERATION_PROMPT = """
[INST]
You are being{agentId} at position ({x}, {y}). {grid_description} You have a summary memory of the situation so far: {memory}.
Based on the above, choose your next move. Respond with only one of the following options, and nothing else: "x+1", "x-1", "y+1", "y-1", or "stay".
Do not provide any explanation or additional text.
[/INST]
"""

# Data Models
class StepRequest(BaseModel):
    steps: int

class DirectMessageRequest(BaseModel):
    sender_id: int
    receiver_id: int
    message: str

class BroadcastMessageRequest(BaseModel):
    sender_id: int
    message: str

# Communication Manager
class CommunicationManager:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client

    async def send_direct_message(self, sender_id: int, receiver_id: int, message: str):
        dnd = await self.redis.hget(f"agent:{receiver_id}", "dnd")
        if dnd and dnd.lower() == "true":
            # Store message for later
            await self.redis.rpush(f"agent:{receiver_id}:inbox", f"DM from Agent {sender_id}: {message}")
            logger.info(f"Message stored for Agent {receiver_id} due to DND mode.")
        else:
            # Deliver message immediately
            await self.redis.rpush(f"agent:{receiver_id}:inbox", f"DM from Agent {sender_id}: {message}")
            logger.info(f"Direct message sent from Agent {sender_id} to Agent {receiver_id}.")

    async def broadcast_message(self, sender_id: int, message: str):
        # Store in global broadcast list
        await self.redis.rpush("global_broadcasts", f"Broadcast from Agent {sender_id}: {message}")
        logger.info(f"Broadcast message sent from Agent {sender_id} to all agents.")

    async def fetch_messages(self, agent_id: int):
        # Fetch direct messages
        dm_key = f"agent:{agent_id}:inbox"
        dms = await self.redis.lrange(dm_key, 0, -1)
        # Fetch global broadcasts
        broadcasts = await self.redis.lrange("global_broadcasts", 0, -1)
        # Combine messages
        all_messages = dms + broadcasts
        return all_messages

    async def clear_messages(self, agent_id: int):
        # Clear direct messages
        dm_key = f"agent:{agent_id}:inbox"
        await self.redis.delete(dm_key)
        # Optionally, clear global broadcasts if needed
        # await self.redis.delete("global_broadcasts")
        logger.info(f"Messages cleared for Agent {agent_id}.")

    async def toggle_dnd(self, agent_id: int):
        current_dnd = await self.redis.hget(f"agent:{agent_id}", "dnd")
        new_dnd = "false" if current_dnd and current_dnd.lower() == "true" else "true"
        await self.redis.hset(f"agent:{agent_id}", "dnd", new_dnd)
        logger.info(f"Agent {agent_id} DND mode set to {new_dnd}.")
        return new_dnd

# Initialize Communication Manager
comm_manager = CommunicationManager(redis)

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
                        movement = None
                        for cmd in valid_commands:
                            if cmd == content_lower:
                                movement = cmd
                                break
                        if not movement:
                            # Try to extract command from response
                            for cmd in valid_commands:
                                if cmd in content_lower:
                                    movement = cmd
                                    break
                        if not movement:
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
    supabase.table("movements").delete().neq("agent_id", -1).execute()
    supabase.table("agents").delete().neq("id", -1).execute()

    agents = [
        {
            "id": i,
            "name": f"Agent-{i}",
            "x": random.randint(0, GRID_SIZE - 1),
            "y": random.randint(0, GRID_SIZE - 1),
            "memory": "",
            "dnd": "false"  # Initialize DND mode as false
        }
        for i in range(NUM_AGENTS)
    ]
    supabase.table("agents").insert(agents).execute()
    for agent in agents:
        await redis.hset(f"agent:{agent['id']}", mapping=agent)
    logger.info("Agents initialized.")
    return agents

async def fetch_nearby_messages(agent, agents):
    nearby_agents = [
        a for a in agents if a["id"] != agent["id"] and chebyshev_distance(agent["x"], agent["y"], a["x"], a["y"]) <= CHEBYSHEV_DISTANCE
    ]
    messages = [await redis.hget(f"agent:{a['id']}", "message") for a in nearby_agents]
    return [m for m in messages if m]

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

    agents = [
        {
            "id": int(agent["id"]),
            "name": agent["name"],
            "x": int(agent["x"]),
            "y": int(agent["y"]),
            "memory": agent.get("memory", "")
        }
        for agent in [await redis.hgetall(f"agent:{i}") for i in range(NUM_AGENTS)]
    ]

    for _ in range(request.steps):
        if stop_signal:
            logger.info("Stopping steps due to stop signal.")
            break

        # Message Generation
        for agent in agents:
            # Fetch nearby and global messages
            nearby_messages = await fetch_nearby_messages(agent, agents)
            global_messages = await redis.lrange("global_broadcasts", 0, -1)
            all_messages = nearby_messages + global_messages
            
            # Generate message using LLM
            message_result = await send_llm_request(
                construct_prompt(
                    MESSAGE_GENERATION_PROMPT,
                    agent,
                    all_messages
                )
            )
            if "message" not in message_result:
                logger.warning(f"Skipping message update for Agent {agent['id']} due to invalid response.")
                continue
            await redis.hset(f"agent:{agent['id']}", "message", message_result.get("message", ""))
            await asyncio.sleep(REQUEST_DELAY)

        if stop_signal:
            logger.info("Stopping after message generation due to stop signal.")
            break

        # Memory Generation
        for agent in agents:
            # Fetch nearby and global messages for memory construction
            nearby_messages = await fetch_nearby_messages(agent, agents)
            global_messages = await redis.lrange("global_broadcasts", 0, -1)
            all_messages = nearby_messages + global_messages
            
            # Generate memory using LLM
            memory_result = await send_llm_request(
                construct_prompt(
                    MEMORY_GENERATION_PROMPT,
                    agent,
                    all_messages
                )
            )
            if "memory" not in memory_result:
                logger.warning(f"Skipping memory update for Agent {agent['id']} due to invalid response.")
                continue
            await redis.hset(f"agent:{agent['id']}", "memory", memory_result.get("memory", agent["memory"]))
            await asyncio.sleep(REQUEST_DELAY)

        if stop_signal:
            logger.info("Stopping after memory generation due to stop signal.")
            break

        # Movement Generation
        for agent in agents:
            # Fetch nearby and global messages to inform movement
            nearby_messages = await fetch_nearby_messages(agent, agents)
            global_messages = await redis.lrange("global_broadcasts", 0, -1)
            all_messages = nearby_messages + global_messages

            movement_result = await send_llm_request(
                construct_prompt(
                    MOVEMENT_GENERATION_PROMPT,
                    agent,
                    all_messages
                )
            )
            if "movement" not in movement_result:
                logger.warning(f"Skipping movement update for Agent {agent['id']} due to invalid response.")
                continue

            # Apply movement logic
            movement = movement_result.get("movement", "stay").strip().lower()
            initial_position = (agent["x"], agent["y"])

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
                continue  # No update needed for stay
            else:
                logger.warning(f"Invalid movement command for Agent {agent['id']}: {movement}")
                continue  # Skip invalid commands

            # Log and update position
            logger.info(f"Agent {agent['id']} moved from {initial_position} to ({agent['x']}, {agent['y']}) with action '{movement}'.")
            await redis.hset(f"agent:{agent['id']}", mapping={"x": agent["x"], "y": agent["y"]})
            await asyncio.sleep(REQUEST_DELAY)

    return JSONResponse({"status": f"Performed {request.steps} step(s)."})

@app.post("/stop")
async def stop_simulation():
    global stop_signal
    stop_signal = True
    logger.info("Stop signal triggered.")
    return JSONResponse({"status": "Simulation stopping."})

# Communication Endpoints

@app.post("/toggle_dnd/{agent_id}")
async def toggle_dnd(agent_id: int):
    try:
        new_dnd = await comm_manager.toggle_dnd(agent_id)
        return JSONResponse({"status": f"Agent {agent_id} DND mode set to {new_dnd}."})
    except Exception as e:
        logger.error(f"Error toggling DND for Agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to toggle DND mode.")

@app.post("/send_direct_message")
async def send_direct_message(request: DirectMessageRequest):
    try:
        # Check if sender exists
        sender = await redis.hgetall(f"agent:{request.sender_id}")
        if not sender:
            raise HTTPException(status_code=404, detail="Sender not found.")
        # Check if receiver exists
        receiver = await redis.hgetall(f"agent:{request.receiver_id}")
        if not receiver:
            raise HTTPException(status_code=404, detail="Receiver not found.")
        await comm_manager.send_direct_message(
            sender_id=request.sender_id,
            receiver_id=request.receiver_id,
            message=request.message
        )
        return JSONResponse({"status": "Message sent successfully."})
    except Exception as e:
        logger.error(f"Error sending direct message: {e}")
        raise HTTPException(status_code=500, detail="Failed to send direct message.")

@app.post("/broadcast_message")
async def broadcast_message(request: BroadcastMessageRequest):
    try:
        # Check if sender exists
        sender = await redis.hgetall(f"agent:{request.sender_id}")
        if not sender:
            raise HTTPException(status_code=404, detail="Sender not found.")
        await comm_manager.broadcast_message(
            sender_id=request.sender_id,
            message=request.message
        )
        return JSONResponse({"status": "Broadcast message sent successfully."})
    except Exception as e:
        logger.error(f"Error broadcasting message: {e}")
        raise HTTPException(status_code=500, detail="Failed to broadcast message.")

@app.get("/fetch_messages/{agent_id}")
async def fetch_messages(agent_id: int):
    try:
        # Check if agent exists
        agent = await redis.hgetall(f"agent:{agent_id}")
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found.")
        messages = await comm_manager.fetch_messages(agent_id)
        return JSONResponse({"messages": messages})
    except Exception as e:
        logger.error(f"Error fetching messages for Agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch messages.")

@app.post("/clear_messages/{agent_id}")
async def clear_messages(agent_id: int):
    try:
        # Check if agent exists
        agent = await redis.hgetall(f"agent:{agent_id}")
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found.")
        await comm_manager.clear_messages(agent_id)
        return JSONResponse({"status": f"Agent {agent_id} messages cleared."})
    except Exception as e:
        logger.error(f"Error clearing messages for Agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear messages.")