import os
from dotenv import load_dotenv
import logging
from models import SimulationSettings, PromptSettings

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
GRID_SIZE = 30
NUM_ENTITIES = 10
MAX_STEPS = 100
CHEBYSHEV_DISTANCE = 5
LLM_MODEL = "llama-3.1-8b-instant"
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE = 0.7
REQUEST_DELAY = 1.5
MAX_CONCURRENT_REQUESTS = 1

# Prompt Templates
GRID_DESCRIPTION = "The field size is 30 x 30 with periodic boundary conditions, and there are a total of 10 entities. You are free to move around the field and converse with other entities. Work collectively to solve problems."

DEFAULT_MESSAGE_GENERATION_PROMPT = """
You are entity{entityId} at position ({x}, {y}). {grid_description} You have a summary memory of the situation so far: {memory}. You received messages from the surrounding entities: {messages}. Based on this, choose how to communicate with the surrounding entities. Your message will reach entities up to distance {distance} away. What message do you send?
Respond with only the message content, and nothing else.
"""

DEFAULT_MEMORY_GENERATION_PROMPT = """
You are entity{entityId} at position ({x}, {y}). {grid_description} You have a summary memory of the situation so far: {memory}. You received messages from the surrounding entities: {messages}. Based on this, create a summary of the situation for yourself.
Respond with only the summary, and nothing else.
"""

DEFAULT_MOVEMENT_GENERATION_PROMPT = """
You are entity{entityId} at position ({x}, {y}). {grid_description} You have a summary memory of the situation so far: {memory}.
Based on this, decide your next move. Respond with only one of the following options, and nothing else: "x+1", "x-1", "y+1", "y-1", or "stay".
Do not provide any explanation or additional text.
"""

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simulation_app")

def add_log(message: str):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    print(formatted_message)  # Log to console or store as needed
