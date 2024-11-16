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

load_dotenv()

# Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
REDIS_ENDPOINT = "cute-crawdad-25113.upstash.io"
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# Simulation Configuration
GRID_SIZE = 30  # Updated to 30x30 grid
NUM_AGENTS = 10
MAX_STEPS = 100
CHEBYSHEV_DISTANCE = 5
LLM_MODEL = "llama-3.2-11b-vision-preview"
LLM_MAX_TOKENS = 2048
LLM_TEMPERATURE = 0.7
REQUEST_DELAY = 2.1  # Fixed delay in seconds between requests to control pace
MAX_CONCURRENT_REQUESTS = 1  # Only one concurrent request

# Redis & Supabase Initialization
# Initialize Redis client with TLS
redis = Redis(
    host="cute-crawdad-25113.upstash.io",
    port=6379,
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True,
    ssl=True  # Enable SSL/TLS
)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# FastAPI Application
app = FastAPI()

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simulation_app")

# Semaphore for throttling concurrent requests
global_request_semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)

# Prompt Templates
GRID_DESCRIPTION = "The field size is 30 x 30 with periodic boundary conditions, and there are a total of 10 entities. You are free to move around the field and converse with other entities."

MESSAGE_GENERATION_PROMPT = """
[INST]
You are entity{agentId} at position ({x}, {y}). {grid_description} You have a summary memory of the situation so far: {memory}. You received messages from the surrounding entities: {messages}. Based on the above, you send a message to the surrounding entities. Your message will reach entities up to distance {distance} away. What message do you send? [/INST]
"""

MEMORY_GENERATION_PROMPT = """
[INST]
You are entity{agentId} at position ({x}, {y}). {grid_description} You have a summary memory of the situation so far: {memory}. You received messages from the surrounding entities: {messages}. Based on the above, summarize the situation you and the other entities have been in so far for you to remember. [/INST]
"""

MOVEMENT_GENERATION_PROMPT = """
[INST]
You are entity{agentId} at position ({x}, {y}). {grid_description} You have a summary memory of the situation so far: {memory}. Based on the above, what is your next move command? Choose only one of the following: ["x+1", "x-1", "y+1", "y-1", "stay"] [/INST]
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

async def send_llm_request(prompt):
    async with global_request_semaphore:  # Limits concurrent requests to 1
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
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(GROQ_API_ENDPOINT, headers=headers, json=body)
                if response.status_code == 429:
                    logger.error("Received 429 Too Many Requests.")
                    return {"message": "", "memory": "", "movement": "stay"}  # Fallback
                response.raise_for_status()
                result = response.json()

                # Validate expected keys
                if not all(key in result for key in ["choices"]):
                    logger.warning(f"Incomplete response from LLM: {result}")
                    return {"message": "", "memory": "", "movement": "stay"}  # Fallback

                # Extract content from choices
                content = result["choices"][0]["message"]["content"]

                # Depending on the prompt, categorize the response
                if "What message do you send?" in prompt:
                    await asyncio.sleep(REQUEST_DELAY)  # Wait after request
                    return {"message": content.strip()}
                elif "summarize the situation" in prompt:
                    await asyncio.sleep(REQUEST_DELAY)  # Wait after request
                    return {"memory": content.strip()}
                elif "what is your next move command?" in prompt:
                    await asyncio.sleep(REQUEST_DELAY)  # Wait after request
                    return {"movement": content.strip()}
                else:
                    logger.warning(f"Unexpected prompt type: {prompt}")
                    await asyncio.sleep(REQUEST_DELAY)  # Wait after request
                    return {"message": "", "memory": "", "movement": "stay"}

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.error("Received 429 Too Many Requests.")
                else:
                    logger.error(f"HTTP error during LLM request: {e}")
                await asyncio.sleep(REQUEST_DELAY)  # Wait before next attempt
                return {"message": "", "memory": "", "movement": "stay"}  # Fallback
            except Exception as e:
                logger.error(f"Unexpected error during LLM request: {e}")
                await asyncio.sleep(REQUEST_DELAY)  # Wait before next attempt
                return {"message": "", "memory": "", "movement": "stay"}  # Fallback

async def initialize_agents():
    # Clear dependent records in the "movements" table
    logger.info("Deleting dependent records from 'movements' table.")
    supabase.table("movements").delete().neq("agent_id", -1).execute()

    # Clear the "agents" table
    logger.info("Deleting records from 'agents' table.")
    supabase.table("agents").delete().neq("id", -1).execute()

    # Initialize new agents with the required fields
    logger.info("Creating new agents.")
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

    supabase.table("agents").insert(agents).execute()

    # Update Redis with agent data
    for agent in agents:
        await redis.hset(f"agent:{agent['id']}", mapping=agent)
    logger.info("Agents initialized successfully.")
    return agents

async def fetch_nearby_messages(agent, agents):
    nearby_agents = [a for a in agents if a["id"] != agent["id"] and chebyshev_distance(agent["x"], agent["y"], a["x"], a["y"]) <= CHEBYSHEV_DISTANCE]
    messages = [await redis.hget(f"agent:{a['id']}", "message") for a in nearby_agents]
    return [m for m in messages if m]

# Simulation API Endpoints
@app.post("/reset")
async def reset_simulation():
    await redis.flushdb()
    agents = await initialize_agents()
    return JSONResponse({"status": "Simulation reset successfully.", "agents": agents})

@app.post("/start")
async def start_simulation():
    agents = await initialize_agents()
    return JSONResponse({"status": "Simulation started successfully.", "agents": agents})

@app.post("/step")
async def perform_steps(request: StepRequest):
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
        # Message Generation
        message_tasks = [
            send_llm_request(
                construct_prompt(
                    MESSAGE_GENERATION_PROMPT,
                    agent,
                    await fetch_nearby_messages(agent, agents)
                )
            )
            for agent in agents
        ]
        message_results = await asyncio.gather(*message_tasks)

        for agent, message_result in zip(agents, message_results):
            if "message" not in message_result:
                logger.warning(f"Skipping message update for Agent {agent['id']} due to invalid response.")
                continue
            await redis.hset(f"agent:{agent['id']}", "message", message_result.get("message", ""))

        # Memory Generation
        movement_tasks = [
            send_llm_request(construct_prompt(MOVEMENT_GENERATION_PROMPT, agent, []))
            for agent in agents
        ]
        movement_results = await asyncio.gather(*movement_tasks)

        for agent, movement_result in zip(agents, movement_results):
            if "movement" not in movement_result:
                logger.warning(f"Skipping movement update for Agent {agent['id']} due to invalid response.")
                continue
            
            # Apply movement logic
            movement = movement_result.get("movement", "stay")
            initial_position = (agent["x"], agent["y"])  # Log initial position
            if movement == "x+1":
                agent["x"] = (agent["x"] + 1) % GRID_SIZE
            elif movement == "x-1":
                agent["x"] = (agent["x"] - 1) % GRID_SIZE
            elif movement == "y+1":
                agent["y"] = (agent["y"] + 1) % GRID_SIZE
            elif movement == "y-1":
                agent["y"] = (agent["y"] - 1) % GRID_SIZE

            # Log the movement action and updated position
            logger.info(f"Agent {agent['id']} moved from {initial_position} to ({agent['x']}, {agent['y']}) with action '{movement}'.")

            # Update Redis with new position
            await redis.hset(f"agent:{agent['id']}", mapping=agent)

    return JSONResponse({"status": f"Performed {request.steps} step(s)."})

@app.post("/stop")
async def stop_simulation():
    return JSONResponse({"status": "Simulation stopped successfully."})
