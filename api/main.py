# src/main.py

import os
import random
import json
import asyncio
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from supabase import create_client, Client
import httpx
import logging
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ---------------------- Configuration and Initialization ----------------------

# Load environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_ENDPOINT = os.getenv("GROQ_API_ENDPOINT")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AUTH_TOKEN = os.getenv("AUTH_TOKEN")
E2B_API_KEY = os.getenv("E2B_API_KEY")  # Not used in current implementation

if not all([SUPABASE_URL, SUPABASE_KEY, GROQ_API_ENDPOINT, GROQ_API_KEY, AUTH_TOKEN]):
    raise EnvironmentError("One or more required environment variables are missing.")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize FastAPI app
app = FastAPI(title="LLM-Based Multi-Agent Simulation")

# Initialize Rate Limiter (Moderate limit: 60 requests per minute)
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Setup logging
logger = logging.getLogger("simulation_app")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '[%(asctime)s] [%(levelname)s] %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# Simulation Configuration
GRID_SIZE = 30
NUM_AGENTS = 10
MAX_STEPS = 100
CHEBYSHEV_DISTANCE = 5
LLM_MODEL = "llama-3.2-90b-vision-preview"
LLM_MAX_TOKENS = 2048  # As per user request
LLM_TEMPERATURE = 0.7

# ---------------------- Security Dependency ----------------------

async def verify_auth_token(x_auth_token: str = Header(...)):
    if x_auth_token != AUTH_TOKEN:
        logger.warning("Unauthorized access attempt with token: %s", x_auth_token)
        raise HTTPException(status_code=401, detail="Unauthorized")
    return x_auth_token

# ---------------------- Prompt Templates ----------------------

GRID_DESCRIPTION = (
    "This simulation operates on a 30x30 grid. Agents interact within this environment based "
    "on their positions and interactions with other agents."
)

MESSAGE_GENERATION_PROMPT = """
[INST]  
You are agent{agentId} at position ({x}, {y}).

**Description of the Simulation:**  
{grid_description}

Consider your surroundings, recent experiences, and any memories or thoughts that may be relevant to share with others nearby. If there is something you feel would benefit others or help advance your own interests, decide whether to send a message to agents within reach. Reflect on the potential impact of your message and the purpose behind sharing it.  
[/INST]
"""

MEMORY_GENERATION_PROMPT = """
[INST]  
You are agent{agentId} at position ({x}, {y}).

**Description of the Simulation:**  
{grid_description}

Reflect on your recent experiences and interactions, considering what stands out or holds personal significance. Choose whether to remember certain details or let them fade, prioritizing what you believe might be meaningful for your future decisions or goals. Decide freely what, if anything, you wish to keep in memory.  
[/INST]
"""

MOVEMENT_GENERATION_PROMPT = """
[INST]  
You are agent{agentId} at position ({x}, {y}).

**Description of the Simulation:**  
{grid_description}

Consider your current location, your recent experiences, and any personal goals you may have. Reflect on whether moving is necessary to further your aims or if remaining where you are might be preferable. If you choose to move, select one of the following directions: "x+1", "x-1", "y+1", "y-1". If you see no reason to move, simply remain where you are.  
[/INST]
"""

# ---------------------- Helper Functions ----------------------

def chebyshev_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    dx = min(abs(x1 - x2), GRID_SIZE - abs(x1 - x2))
    dy = min(abs(y1 - y2), GRID_SIZE - abs(y1 - y2))
    return max(dx, dy)

def get_nearby_agents(agent: Dict, agents: List[Dict]) -> List[Dict]:
    nearby = []
    for other in agents:
        if other['id'] == agent['id']:
            continue
        distance = chebyshev_distance(agent['x'], agent['y'], other['x'], other['y'])
        if distance <= CHEBYSHEV_DISTANCE:
            nearby.append(other)
    return nearby

def construct_prompt(template: str, agent: Dict) -> str:
    return template.format(
        agentId=agent['id'],
        x=agent['x'],
        y=agent['y'],
        grid_description=GRID_DESCRIPTION
    )

def parse_llm_response(response: str) -> Dict:
    try:
        parsed = json.loads(response)
        return {
            "message": parsed.get("message", ""),
            "memory": parsed.get("memory", ""),
            "movement": parsed.get("movement", "stay")
        }
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM response: %s", response)
        return {
            "message": "",
            "memory": "",
            "movement": "stay"
        }

async def send_llm_request(prompt: str) -> Optional[Dict]:
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {GROQ_API_KEY}'
    }
    body = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": LLM_MAX_TOKENS
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(GROQ_API_ENDPOINT, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
            # Adjust parsing based on Groq Cloud's actual response structure
            # Assuming similar to OpenAI's API for demonstration
            llm_output = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            return parse_llm_response(llm_output)
        except httpx.HTTPError as e:
            logger.error("LLM Request failed: %s", e)
            return None

def initialize_agents() -> List[Dict]:
    agents = []
    try:
        supabase.table("agents").delete().neq("id", -1).execute()  # Clear existing agents
        for i in range(NUM_AGENTS):
            agent = {
                "id": i,
                "name": f"agent{i}",
                "x": random.randint(0, GRID_SIZE - 1),
                "y": random.randint(0, GRID_SIZE - 1),
                "memory": "No memory"
            }
            agents.append(agent)
            # Insert into Supabase
            supabase.table("agents").insert({
                "id": agent["id"],
                "name": agent["name"],
                "x": agent["x"],
                "y": agent["y"],
                "memory": agent["memory"]
            }).execute()
        logger.info("Initialized Agents:")
        for agent in agents:
            logger.info(agent)
    except Exception as e:
        logger.error("Failed to initialize agents: %s", e)
        raise
    return agents

def update_agent_position(agent: Dict, movement: str):
    if movement == "x+1":
        agent["x"] = (agent["x"] + 1) % GRID_SIZE
    elif movement == "x-1":
        agent["x"] = (agent["x"] - 1) % GRID_SIZE
    elif movement == "y+1":
        agent["y"] = (agent["y"] + 1) % GRID_SIZE
    elif movement == "y-1":
        agent["y"] = (agent["y"] - 1) % GRID_SIZE
    elif movement == "stay":
        pass  # No movement
    else:
        logger.warning("Invalid movement command '%s' for agent %s. Staying in place.", movement, agent['id'])

async def perform_step(step: int):
    agents = fetch_all_agents()
    agent_responses = {}

    for agent in agents:
        received_messages = fetch_messages(step, agent, agents)
        message_prompt = construct_prompt(MESSAGE_GENERATION_PROMPT, agent)
        memory_prompt = construct_prompt(MEMORY_GENERATION_PROMPT, agent)
        movement_prompt = construct_prompt(MOVEMENT_GENERATION_PROMPT, agent)

        # Generate Message
        message_response = await send_llm_request(message_prompt)
        message_content = message_response["message"] if message_response else ""

        if message_content:
            try:
                supabase.table("messages").insert({
                    "step": step,
                    "sender_id": agent["id"],
                    "content": message_content
                }).execute()
                logger.info("Agent %s sent message: %s", agent["id"], message_content)
            except Exception as e:
                logger.error("Failed to insert message for agent %s: %s", agent["id"], e)

        # Generate Memory
        memory_response = await send_llm_request(memory_prompt)
        updated_memory = memory_response["memory"] if memory_response else agent["memory"]

        try:
            supabase.table("agents").update({
                "memory": updated_memory
            }).eq("id", agent["id"]).execute()
            logger.info("Agent %s updated memory.", agent["id"])
        except Exception as e:
            logger.error("Failed to update memory for agent %s: %s", agent["id"], e)

        # Generate Movement
        movement_response = await send_llm_request(movement_prompt)
        movement = movement_response["movement"] if movement_response else "stay"

        agent_responses[agent["id"]] = movement

    # After all agents have responded, handle movements
    for agent in agents:
        movement = agent_responses.get(agent["id"], "stay")
        update_agent_position(agent, movement)
        try:
            # Update position in Supabase
            supabase.table("agents").update({
                "x": agent["x"],
                "y": agent["y"]
            }).eq("id", agent["id"]).execute()
            # Insert movement into Supabase
            supabase.table("movements").insert({
                "step": step,
                "agent_id": agent["id"],
                "movement": movement
            }).execute()
            logger.info("Agent %s moved %s to (%s, %s)", agent["id"], movement, agent["x"], agent["y"])
        except Exception as e:
            logger.error("Failed to update movement for agent %s: %s", agent["id"], e)

def fetch_all_agents() -> List[Dict]:
    try:
        response = supabase.table("agents").select("*").execute()
        return response.data
    except Exception as e:
        logger.error("Failed to fetch agents: %s", e)
        return []

def fetch_messages(step: int, agent: Dict, agents: List[Dict]) -> List[str]:
    nearby_agents = get_nearby_agents(agent, agents)
    messages = []
    for nearby in nearby_agents:
        try:
            # Fetch the latest message from this nearby agent up to the previous step
            response = supabase.table("messages") \
                .select("content") \
                .eq("sender_id", nearby["id"]) \
                .lte("step", step - 1) \
                .order("step", desc=True) \
                .limit(1) \
                .execute()
            if response.data:
                messages.append(response.data[0]["content"])
        except Exception as e:
            logger.error("Failed to fetch messages for agent %s from agent %s: %s", agent["id"], nearby["id"], e)
    if not messages:
        messages.append("No Messages")
    return messages

def get_simulation_status() -> Dict:
    try:
        response = supabase.table("simulation").select("*").eq("id", 1).execute()
        if response.data:
            return response.data[0]
        else:
            # Initialize simulation status if not present
            supabase.table("simulation").insert({
                "id": 1,
                "current_step": 0,
                "status": "stopped"
            }).execute()
            return {"id": 1, "current_step": 0, "status": "stopped"}
    except Exception as e:
        logger.error("Failed to fetch simulation status: %s", e)
        raise

def update_simulation_status(current_step: int, status: str):
    try:
        supabase.table("simulation").update({
            "current_step": current_step,
            "status": status
        }).eq("id", 1).execute()
        logger.info("Simulation status updated to '%s' at step %s.", status, current_step)
    except Exception as e:
        logger.error("Failed to update simulation status: %s", e)

# ---------------------- API Models ----------------------

class StepRequest(BaseModel):
    steps: Optional[int] = 1  # Number of steps to perform

# ---------------------- API Endpoints ----------------------

@app.post("/reset", dependencies=[Depends(verify_auth_token)])
@limiter.limit("10/minute")  # Example rate limit for /reset
async def reset_simulation(request: Request):
    try:
        # Clear specific tables
        supabase.table("agents").delete().neq("id", -1).execute()
        supabase.table("messages").delete().neq("id", -1).execute()
        supabase.table("movements").delete().neq("id", -1).execute()
        # Reset simulation status
        supabase.table("simulation").update({
            "current_step": 0,
            "status": "stopped"
        }).eq("id", 1).execute()
        logger.info("Simulation has been reset.")
        return JSONResponse(content={"status": "Simulation reset successfully."})
    except Exception as e:
        logger.error("Failed to reset simulation: %s", e)
        raise HTTPException(status_code=500, detail="Failed to reset simulation.")

@app.post("/start", dependencies=[Depends(verify_auth_token)])
@limiter.limit("10/minute")  # Example rate limit for /start
async def start_simulation(request: Request):
    try:
        status = get_simulation_status()
        if status["status"] == "running":
            logger.warning("Attempted to start simulation, but it is already running.")
            raise HTTPException(status_code=400, detail="Simulation is already running.")
        # Initialize agents
        initialize_agents()
        # Update simulation status
        update_simulation_status(current_step=0, status="running")
        logger.info("Simulation has been started.")
        return JSONResponse(content={"status": "Simulation started successfully."})
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error("Failed to start simulation: %s", e)
        raise HTTPException(status_code=500, detail="Failed to start simulation.")

@app.post("/step", dependencies=[Depends(verify_auth_token)])
@limiter.limit("60/minute")  # Example rate limit for /step
async def perform_steps(request: StepRequest):
    try:
        status = get_simulation_status()
        if status["status"] != "running":
            logger.warning("Attempted to perform steps, but simulation is not running.")
            raise HTTPException(status_code=400, detail="Simulation is not running.")
        steps_to_perform = request.steps
        if steps_to_perform < 1:
            logger.warning("Invalid number of steps requested: %s", steps_to_perform)
            raise HTTPException(status_code=400, detail="Number of steps must be at least 1.")
        if status["current_step"] + steps_to_perform > MAX_STEPS:
            steps_to_perform = MAX_STEPS - status["current_step"]
            if steps_to_perform <= 0:
                logger.warning("Maximum number of steps reached.")
                raise HTTPException(status_code=400, detail="Maximum number of steps reached.")
        for _ in range(steps_to_perform):
            current_step = get_simulation_status()["current_step"] + 1
            await perform_step(current_step)
            update_simulation_status(current_step=current_step, status="running")
            if current_step >= MAX_STEPS:
                update_simulation_status(current_step=current_step, status="stopped")
                logger.info("Maximum number of steps reached. Simulation stopped.")
                break
        final_step = get_simulation_status()["current_step"]
        logger.info("Performed %s step(s). Current step: %s.", steps_to_perform, final_step)
        return JSONResponse(content={
            "status": f"Performed {steps_to_perform} step(s).",
            "current_step": final_step,
            "max_steps": MAX_STEPS
        })
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error("Failed to perform steps: %s", e)
        raise HTTPException(status_code=500, detail="Failed to perform steps.")

@app.post("/stop", dependencies=[Depends(verify_auth_token)])
@limiter.limit("10/minute")  # Example rate limit for /stop
async def stop_simulation():
    try:
        status = get_simulation_status()
        if status["status"] != "running":
            logger.warning("Attempted to stop simulation, but it is not running.")
            raise HTTPException(status_code=400, detail="Simulation is not running.")
        # Update simulation status
        update_simulation_status(current_step=status["current_step"], status="stopped")
        logger.info("Simulation has been stopped.")
        return JSONResponse(content={"status": "Simulation stopped successfully."})
    except Exception as e:
        logger.error("Failed to stop simulation: %s", e)
        raise HTTPException(status_code=500, detail="Failed to stop simulation.")

# ---------------------- Run the App ----------------------

# To run the app locally for testing, use the command:
# uvicorn src.main:app --host 0.0.0.0 --port 8000

# On Vercel, deploy this script as a serverless function following Vercel's deployment guidelines.
