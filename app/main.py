import asyncio
import os
import random
from typing import List
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from redis.asyncio import Redis
from supabase import create_client, Client
import httpx
import logging
from logging import basicConfig, getLogger
import json
import queue
import threading
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import logfire
import time

# Load environment variables
load_dotenv()

# Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
REDIS_ENDPOINT = os.getenv("REDIS_ENDPOINT")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
LOGFIRE_API_KEY = os.getenv("LOGFIRE_API_KEY")

# GROQ API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# Simulation Configuration (Initial Defaults)
GRID_SIZE = 15  
NUM_ENTITIES = 3
MAX_STEPS = 100
CHEBYSHEV_DISTANCE = 5
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_MAX_TOKENS = 2048
LLM_TEMPERATURE = 0.7
REQUEST_DELAY = 2.2  # Fixed delay in seconds between requests
MAX_CONCURRENT_REQUESTS = 5  # Limit concurrent requests to prevent rate limiting
LOG_LEVEL = "INFO"  # Can be 'DEBUG', 'INFO', 'ERROR', etc.

# Prompt Templates
GRID_DESCRIPTION = f"You are in a {GRID_SIZE} x {GRID_SIZE} field with periodic boundary conditions with {NUM_ENTITIES} other lifeforms. You are free to move around the field and interact with other lifeforms."

DEFAULT_MESSAGE_GENERATION_PROMPT = """
You are lifeform{entityId} at position ({x}, {y}). {grid_description} 
You have a summary memory of the situation so far: {memory}. 
You received messages from the surrounding lifeforms: {messages}. 
Based on the above, you send a message to the surrounding lifeforms. Your message will reach lifeforms up to distance {distance} away. What message do you send?
Respond with only the message content, and nothing else.
"""

DEFAULT_MEMORY_GENERATION_PROMPT = """
You are lifeform{entityId} at position ({x}, {y}). {grid_description} 
You have a summary memory of the situation so far: {memory}. 
You received messages from the surrounding lifeforms: {messages}. 
Based on the above, summarize the situation you and the other lifeforms have been in so far for you to remember.
Respond with only the summary, and nothing else.
"""

DEFAULT_MOVEMENT_GENERATION_PROMPT = """
You are lifeform{entityId} at position ({x}, {y}). {grid_description} 
You have a summary memory of the situation so far: {memory}.
Based on the above, choose your next move. Respond with only one of the following options, and nothing else: "x+1", "x-1", "y+1", "y-1", or "stay".
Do not provide any explanation or additional text.
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

# **Structured Logging with Logfire**
class LogfireHandler(logging.Handler):
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        self.endpoint = "https://logfire.pydantic.dev/cognetsys/cognetics-architect/logs"  # Replace with actual Logfire endpoint

    def emit(self, record):
        log_entry = self.format(record)
        try:
            # Send log to Logfire
            httpx.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                data=log_entry
            )
        except Exception as e:
            print(f"Failed to send log to Logfire: {e}")

# LOGFIRE CONFIG
logfire.configure(service_name="CogNetics Architect")  

# Configure Logger
logger = logging.getLogger('simulation_app')
logger.setLevel(LOG_LEVEL)
logger.propagate = False  # Prevent duplicate logs

# Logfire Handler
if LOGFIRE_API_KEY:
    logfire_handler = LogfireHandler(LOGFIRE_API_KEY)
    logfire_formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "module": "%(module)s", "line": "%(lineno)d"}'
    )
    logfire_handler.setFormatter(logfire_formatter)
    logger.addHandler(logfire_handler)
else:
    print("LOGFIRE_API_KEY not found. Skipping Logfire integration.")

# **Additional Handlers (Optional)**
# You can add other handlers like console or file handlers if needed
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter(
    '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "module": "%(module)s", "line": "%(lineno)d"}'
)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Redis & Supabase Initialization
redis = Redis(
    host=REDIS_ENDPOINT,
    port=6379,
    password=REDIS_PASSWORD,
    decode_responses=True,
    ssl=True  # Enable SSL/TLS
)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# WebSocket client connection listener
connected_clients = []

def send_log_message(message: str):
    """
    Send log message to all connected WebSocket clients.
    """
    for client in connected_clients:
        try:
            asyncio.create_task(client.send_text(message))  # Send log message to client
        except Exception as e:
            logfire.error(f"Error sending message to WebSocket client: {e}")

# Stop signal
stop_signal = False

# Chebyshev Distance Helper Function
def chebyshev_distance(x1, y1, x2, y2):
    return max(abs(x1 - x2), abs(y1 - y2))

async def initialize_entities():
    logfire.info('Resetting simulation state.')
    supabase.table('movements').delete().neq('entity_id', -1).execute()
    supabase.table('entities').delete().neq('id', -1).execute()

    entities = [
        {
            'id': i,
            'name': f'Entity-{i}',
            'x': random.randint(0, GRID_SIZE - 1),
            'y': random.randint(0, GRID_SIZE - 1),
            'memory': ''
        }
        for i in range(NUM_ENTITIES)
    ]
    supabase.table('entities').insert(entities).execute()
    for entity in entities:
        await redis.hset(f'entity:{entity["id"]}', mapping=entity)
    logfire.info('Entities initialized.')
    return entities

async def fetch_nearby_messages(entity, entities):
    nearby_entities = [
        a for a in entities if a['id'] != entity['id'] and chebyshev_distance(entity['x'], entity['y'], a['x'], a['y']) <= CHEBYSHEV_DISTANCE
    ]
    messages = [await redis.hget(f'entity:{a["id"]}', 'message') for a in nearby_entities]
    return [m for m in messages if m]

# Helper function to fetch Prompts from FastAPI
async def fetch_prompts_from_fastapi():
    async with httpx.AsyncClient() as client:
        response = await client.get('http://localhost:8000/api/prompts')
        if response.status_code == 200:
            return response.json()  # Return the fetched prompts
        else:
            logfire.warning('Failed to fetch prompts, using default ones.')
            return {}  # Return an empty dict to trigger the default prompts

from app.endpoints.api import router as api_router

# Lifespan Context Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    global stop_signal
    logfire.info('Starting application lifespan...')
    try:
        # Startup logic
        logfire.info('Application started!')
        await redis.ping()
        logfire.info('Redis connection established.')       
        yield
    finally:
        # Shutdown logic
        logfire.info('Shutting down application...')
        stop_signal = True  # Ensure simulation stops if running
        await redis.close()
        logfire.info('Redis connection closed.')

# Create FastAPI app with lifespan
app = FastAPI(
    title="CogNetics Architect", 
    version="0.0.3", 
    description="This is the enhanced implementation of the <a href='https://arxiv.org/pdf/2411.03252' target='_blank'>Takata et al experiment.</a>",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router, prefix='/api', tags=['simulation'])

# **Inspector Integration (Assuming a Monitoring Tool)**
# Example: Using FastAPI's built-in middleware or a third-party tool like Prometheus
# Here, we'll add a simple middleware for request inspection

from fastapi.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["*"]
)

# Additional inspectors can be integrated here as needed
