from core import app
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from utils import add_log, LOG_QUEUE, logger
from endpoints.database import redis, supabase
from typing import Optional
from world import World  # Import the World class
from config import (
    GRID_SIZE,
    NUM_ENTITIES,
    REDIS_ENDPOINT,
    REDIS_PASSWORD,
)
from redis.asyncio import Redis
import random
import asyncio

# Initialize Router
router = APIRouter()

# Stop signal for simulation control
stop_signal = False

# Endpoint to create and summarize a test world
@router.post("/test", tags=["Worlds"])
async def test_world(
    world_id: Optional[int] = 1,
    grid_size: Optional[int] = 30,
    num_agents: Optional[int] = 10,
    num_tasks: Optional[int] = 100,
    total_resources: Optional[int] = 1000,
    consumed_resources: Optional[int] = 800
):
    """
    Create a test world with specified parameters and summarize its state.
    """
    try:
        # Generate test tasks
        tasks = [{"id": i, "progress": 0.8, "duration": 5} for i in range(num_tasks)]
        
        # Generate resource distribution
        resources = {
            "total": total_resources,
            "consumed": consumed_resources,
            "distribution": {i: total_resources // num_agents for i in range(num_agents)}
        }
        
        # Create the world instance
        test_world = World(
            world_id=world_id,
            grid_size=grid_size,
            num_agents=num_agents,
            tasks=tasks,
            resources=resources
        )
        
        # Summarize the world's state
        summary = test_world.summarize_state()
        add_log(f"World {world_id} Summary: {summary}")
        
        # Return the summary as a JSON response
        return JSONResponse({
            "status": "Test world summarized successfully.",
            "world_id": world_id,
            "summary": summary
        })
    except Exception as e:
        error_message = f"Error testing world: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@router.post("/adjust_worlds", tags=["Worlds"])
async def adjust_worlds(new_world_count: int):
    current_count = len(app.state.worlds)
    if new_world_count > current_count:
        for i in range(current_count, new_world_count):
            app.state.worlds.append({
                "world_id": i,
                "settings": {
                    "task_priorities": {"default_task": 1.0},
                    "agent_behaviors": {"cooperation": 0.5},
                    "resource_allocation": {"default_resource": 100},
                    "environment_changes": {"scarcity_level": 0.2},
                },
            })
    elif new_world_count < current_count:
        app.state.worlds = app.state.worlds[:new_world_count]

    return {"message": f"World count adjusted to {new_world_count}"}
