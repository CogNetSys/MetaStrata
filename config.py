import os
from dotenv import load_dotenv
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
LLM_MODEL = "llama3-8b-8192"
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE = 0.7
REQUEST_DELAY = 2.3
MAX_CONCURRENT_REQUESTS = 4

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
Based on this, decide your next move. Please choose one of the following options, and nothing else: "x+1", "x-1", "y+1", "y-1", or "stay".
Do not provide any explanation or additional text.
"""

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
