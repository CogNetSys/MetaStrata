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

load_dotenv()

# Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
REDIS_ENDPOINT = os.getenv("REDIS_ENDPOINT")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# Simulation Configuration
GRID_SIZE = 30  # Updated to 30x30 grid
NUM_AGENTS = 10
MAX_STEPS = 100
CHEBYSHEV_DISTANCE = 5
LLM_MODEL = "llama-3.2-90b-vision-preview"
LLM_MAX_TOKENS = 2048
LLM_TEMPERATURE = 0.7
REQUEST_DELAY = 1.0  # Delay in seconds between requests to control pace

# Redis & Supabase Initialization
redis = Redis(host=REDIS_ENDPOINT, password=REDIS_PASSWORD)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# FastAPI Application
app = FastAPI()

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simulation_app")

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
    messages_str = "\n".join(messages)
    return template.format(
        agentId=agent["id"], x=agent["x"], y=agent["y"],
        grid_description=GRID_DESCRIPTION, memory=agent.get("memory", ""),
        messages=messages_str, distance=CHEBYSHEV_DISTANCE
    )

async def send_llm_request(prompt):
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
        response = await client.post(GROQ_API_ENDPOINT, headers=headers, json=body)
        response.raise_for_status()
        return response.json()

async def initialize_agents():
    agents = [{"id": i, "x": random.randint(0, GRID_SIZE - 1), "y": random.randint(0, GRID_SIZE - 1), "memory": ""} for i in range(NUM_AGENTS)]
    for agent in agents:
        await redis.hset(f"agent:{agent['id']}", mapping=agent)
    await supabase.table("agents").delete().neq("id", -1).execute()
    await supabase.table("agents").insert(agents).execute()
    return agents

async def fetch_nearby_messages(agent, agents):
    nearby_agents = [a for a in agents if a["id"] != agent["id"] and chebyshev_distance(agent["x"], agent["y"], a["x"], a["y"]) <= CHEBYSHEV_DISTANCE]
    messages = [await redis.hget(f"agent:{a['id']}", "message") for a in nearby_agents]
    return [m for m in messages if m]

async def paced_request_execution(tasks):
    """
    Process tasks with a delay between each request to avoid overloading the API.
    """
    results = []
    for task in tasks:
        result = await task
        results.append(result)
        await asyncio.sleep(REQUEST_DELAY)  # Delay between requests
    return results

# Simulation API Endpoints
@app.post("/reset")
async def reset_simulation():
    await redis.flushdb()
    await initialize_agents()
    return JSONResponse({"status": "Simulation reset successfully."})

@app.post("/start")
async def start_simulation():
    agents = await initialize_agents()
    return JSONResponse({"status": "Simulation started successfully.", "agents": agents})

@app.post("/step")
async def perform_steps(request: StepRequest):
    agents = [await redis.hgetall(f"agent:{i}") for i in range(NUM_AGENTS)]
    for _ in range(request.steps):
        # Message Generation
        message_tasks = [send_llm_request(construct_prompt(MESSAGE_GENERATION_PROMPT, agent, await fetch_nearby_messages(agent, agents))) for agent in agents]
        message_results = await paced_request_execution(message_tasks)
        
        for agent, message_result in zip(agents, message_results):
            await redis.hset(f"agent:{agent['id']}", "message", message_result.get("message", ""))

        # Memory Generation
        memory_tasks = [send_llm_request(construct_prompt(MEMORY_GENERATION_PROMPT, agent, await fetch_nearby_messages(agent, agents))) for agent in agents]
        memory_results = await paced_request_execution(memory_tasks)
        
        for agent, memory_result in zip(agents, memory_results):
            await redis.hset(f"agent:{agent['id']}", "memory", memory_result.get("memory", agent["memory"]))

        # Movement Generation
        movement_tasks = [send_llm_request(construct_prompt(MOVEMENT_GENERATION_PROMPT, agent, [])) for agent in agents]
        movement_results = await paced_request_execution(movement_tasks)
        
        for agent, movement_result in zip(agents, movement_results):
            movement = movement_result.get("movement", "stay")
            if movement == "x+1": agent["x"] = (agent["x"] + 1) % GRID_SIZE
            elif movement == "x-1": agent["x"] = (agent["x"] - 1) % GRID_SIZE
            elif movement == "y+1": agent["y"] = (agent["y"] + 1) % GRID_SIZE
            elif movement == "y-1": agent["y"] = (agent["y"] - 1) % GRID_SIZE
            await redis.hset(f"agent:{agent['id']}", mapping=agent)

    return JSONResponse({"status": f"Performed {request.steps} step(s)."})

@app.post("/stop")
async def stop_simulation():
    return JSONResponse({"status": "Simulation stopped successfully."})
