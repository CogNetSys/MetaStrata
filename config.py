import os
from dotenv import load_dotenv
import logging

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
MAX_COMMUNICATION_DISTANCE = 5
LLM_MODEL = "llama-3.2-11b-vision-preview"
LLM_MAX_TOKENS = 2048
LLM_TEMPERATURE = 0.7
REQUEST_DELAY = 2.2
MAX_CONCURRENT_REQUESTS = 5
CHEBYSHEV_DISTANCE = 5

# Prompt Templates
GRID_DESCRIPTION = "The simulation occurs on a 2D lattice with dimensions 30 x 30. It uses toroidal (wrap-around) boundary conditions. There are 10 entities on the grid. You can move between adjacent cells (up, down, left, right), and communicate with other entities within a Chebyshev distance of 5 cells. Work collectively to solve problems."

DEFAULT_MESSAGE_GENERATION_PROMPT = """
[INST]
You are entity{entityId} at position ({x}, {y}). {grid_description} You have a summary memory of the situation so far: {memory}. You received messages from the surrounding entities: {messages}. Based on this, choose how to communicate with the surrounding entities. Your message will reach entities up to distance {distance} away. What message do you send?
Respond with only the message content, and nothing else.
[/INST]
"""

DEFAULT_MEMORY_GENERATION_PROMPT = """
[INST]
You are entity{entityId} at position ({x}, {y}). {grid_description} You have a summary memory of the situation so far: {memory}. You received messages from the surrounding entities: {messages}. Based on this, create a summary of the situation for yourself.
Respond with only the summary, and nothing else.
[/INST]
"""

DEFAULT_MOVEMENT_GENERATION_PROMPT = """
[INST]
You are entity{entityId} at position ({x}, {y}). {grid_description} You have a summary memory of the situation so far: {memory}.
Based on this, decide your next move. Respond with only one of the following options, and nothing else: "x+1", "x-1", "y+1", "y-1", or "stay".
Consider the periodic boundary conditions when deciding your move. Do not provide any explanation or additional text.
[/INST]
"""

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simulation_app")