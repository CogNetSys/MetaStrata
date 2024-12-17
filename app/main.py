import os
import random
import asyncio
import time
from typing import List, Dict
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
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
from fastapi.middleware.cors import CORSMiddleware
import json
from datetime import datetime
from fastapi.openapi.docs import get_swagger_ui_html

load_dotenv()

# Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
REDIS_ENDPOINT = "cute-crawdad-25113.upstash.io"
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# Environment Variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# Simulation Configuration (Initial Defaults)
GRID_SIZE = 10  
NUM_ENTITIES = 3
MAX_STEPS = 100
CHEBYSHEV_DISTANCE = 5
LLM_MODEL = "llama-3.1-8b-instant"
LLM_MAX_TOKENS = 2048
LLM_TEMPERATURE = 0.7
REQUEST_DELAY = 2.2  # Fixed delay in seconds between requests
MAX_CONCURRENT_REQUESTS = 1  # Limit concurrent requests to prevent rate limiting

# Prompt Templates
GRID_DESCRIPTION = "You are in a field with periodic boundary conditions with other beings. You are free to move around the field and converse with other beings."

DEFAULT_MESSAGE_GENERATION_PROMPT = """
[INST]
You are being{entityId} at position ({x}, {y}). {grid_description} You have a summary memory of the situation so far: {memory}. You received messages from the surrounding beings: {messages}. Based on the above, you send a message to the surrounding beings. Your message will reach beings up to distance {distance} away. What message do you send?
Respond with only the message content, and nothing else.
[/INST]
"""

DEFAULT_MEMORY_GENERATION_PROMPT = """
[INST]
You are being{entityId} at position ({x}, {y}). {grid_description} You have a summary memory of the situation so far: {memory}. You received messages from the surrounding beings: {messages}. Based on the above, summarize the situation you and the other beings have been in so far for you to remember.
Respond with only the summary, and nothing else.
[/INST]
"""

DEFAULT_MOVEMENT_GENERATION_PROMPT = """
[INST]
You are being{entityId} at position ({x}, {y}). {grid_description} You have a summary memory of the situation so far: {memory}.
Based on the above, choose your next move. Respond with only one of the following options, and nothing else: "x+1", "x-1", "y+1", "y-1", or "stay".
Do not provide any explanation or additional text.
[/INST]
"""

# Define the configuration model
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

# Data Models
class StepRequest(BaseModel):
    steps: int

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

# Stop signal
stop_signal = False

# Chebyshev Distance Helper Function for calculating the distance for Nearby Entities function.
def chebyshev_distance(x1, y1, x2, y2):
    return max(abs(x1 - x2), abs(y1 - y2))

async def initialize_entities():
    logger.info("Resetting simulation state.")
    supabase.table("movements").delete().neq("entity_id", -1).execute()
    supabase.table("entities").delete().neq("id", -1).execute()

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
    supabase.table("entities").insert(entities).execute()
    for entity in entities:
        await redis.hset(f"entity:{entity['id']}", mapping=entity)
    logger.info("Entities initialized.")
    return entities

async def fetch_nearby_messages(entity, entities):
    nearby_entities = [
        a for a in entities if a["id"] != entity["id"] and chebyshev_distance(entity["x"], entity["y"], a["x"], a["y"]) <= CHEBYSHEV_DISTANCE
    ]
    messages = [await redis.hget(f"entity:{a['id']}", "message") for a in nearby_entities]
    return [m for m in messages if m]

# Helper function to fetch Prompts from FastAPI
async def fetch_prompts_from_fastapi():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/api/prompts")
        if response.status_code == 200:
            return response.json()  # Return the fetched prompts
        else:
            logger.warning("Failed to fetch prompts, using default ones.")
            return {}  # Return an empty dict to trigger the default prompts

from app.endpoints.api import router as api_router

# Lifespan Context Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    global stop_signal
    logger.info("Starting application lifespan...")
    try:
        # Startup logic
        await redis.ping()
        logger.info("Redis connection established.")       
        yield
    finally:
        # Shutdown logic
        logger.info("Shutting down application...")
        stop_signal = True  # Ensure simulation stops if running
        await redis.close()
        logger.info("Redis connection closed.")

# Create FastAPI app with lifespan
app = FastAPI(
    title="CogNetics Architect", 
    version="0.0.3", 
    description="This is the enhanced implementation of the <a href='https://arxiv.org/pdf/2411.03252' target='_blank'>Takata et al experiment.</a>",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins or specify your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api", tags=["simulation"])