import os
from dotenv import load_dotenv
from models import SimulationSettings, PromptSettings, World
from typing import List, Dict

# Load environment variables
load_dotenv()

# Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
REDIS_ENDPOINT = "cute-crawdad-25113.upstash.io"
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# Simulation Configuration
NUM_WORLDS = 1
GRID_SIZE = 30
NUM_ENTITIES = 10
MAX_STEPS = 100
CHEBYSHEV_DISTANCE = 5
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE = 0.7
REQUEST_DELAY = 2.3
MAX_CONCURRENT_REQUESTS = 1

# Prompt Templates
GRID_DESCRIPTION = "The field size is 30 x 30 with periodic boundary conditions, and there are a total of 6 entities. You are free to move around the field and converse with other entities. Work collectively to solve problems."

DEFAULT_MESSAGE_GENERATION_PROMPT = """
You are entity{entityId} at position ({x}, {y}). {grid_description} You have a summary memory of the situation so far: {memory}. You received messages from the surrounding entities: {messages}. Based on this, choose how to communicate with the surrounding entities. Your message will reach entities up to distance {distance} away. What message do you send?
Respond with only the message content, and nothing else.
"""

DEFAULT_MEMORY_GENERATION_PROMPT = """
You are entity{entityId} at position ({x}, {y}). {grid_description} You have a summary memory of the situation so far: {memory}. You received messages from the surrounding entities: {messages}. Based on this, create a very brief summary of the situation for yourself.
Respond with only the summary, and nothing else.
"""

DEFAULT_MOVEMENT_GENERATION_PROMPT = """
You are entity{entityId} at position ({x}, {y}). {grid_description} You have a summary memory of the situation so far: {memory}.
Based on this, decide your next move. Please choose one of the following options, and nothing else: "x+1", "x-1", "y+1", "y-1", or "stay".
Respond with a single action corresponding to your next move. No extra text.
"""

# Number of Worlds
NUM_WORLDS = int(os.getenv("NUM_WORLDS", 1))  # Default to 1 if not set

# mTNN API Endpoint
#MTNN_API_ENDPOINT = os.getenv("MTNN_API_ENDPOINT")
MTNN_API_ENDPOINT = "http://localhost:8000/mtnn/submit_summary"

# Environment variables for logs
LOG_DIR = os.getenv("LOG_DIR", "/var/log/myapp")  # Default directory if not specified
LOG_FILE_NAME = os.getenv("LOG_FILE", "simulation_logs.log")  # Default log file name

# Combine the directory and file name
LOG_FILE = os.path.join(LOG_DIR, LOG_FILE_NAME)

# Log Queue Max Size (for the in-memory log queue)
LOG_QUEUE_MAX_SIZE = int(os.getenv("LOG_QUEUE_MAX_SIZE", 100))  # Default to 100 if not set

# Set Log Level from environment, defaulting to INFO if not set
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Import the necessary utilities only where needed, not at the top
def get_logger_and_log_queue():
    from utils import add_log, LOG_QUEUE, logger
    return add_log, LOG_QUEUE, logger

def create_world_config(world_id: int, grid_size: int, num_agents: int, tasks: list, resources: dict) -> dict:
    """
    Create a configuration dictionary for initializing a world.
    """
    return {
        "world_id": world_id,
        "grid_size": grid_size,
        "num_agents": num_agents,
        "tasks": tasks,  # Include tasks
        "resources": resources  # Include resources
    }
