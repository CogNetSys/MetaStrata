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

# Helper function to initialize entities
async def initialize_entities():
    logger.info("Resetting simulation state.")
    await redis.flushdb()  # Clear all Redis data, including entity_keys

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

    for entity in entities:
        entity_key = f"entity:{entity['id']}"
        await redis.hset(entity_key, mapping=entity)
        await redis.lpush("entity_keys", entity_key)  # Add to entity_keys list
        await redis.delete(f"{entity['id']}:messages")  # Clear message queue

    logger.info("Entities initialized.")
    return entities

# Endpoint to create and summarize a test world
@router.post("/test", tags=["World Simulation"])
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

# Endpoint to initialize and summarize a real simulation world
@router.post("/initialize_and_summarize", tags=["World Simulation"])
async def initialize_and_summarize_world():
    """
    Initialize a real-world simulation and summarize its state.
    """
    try:
        # Initialize entities for the simulation
        entities = await initialize_entities()
        
        # Convert entities into a format compatible with the World class
        tasks = [{"id": entity["id"], "progress": 0.5, "duration": 10} for entity in entities]
        resources = {
            "total": 5000,
            "consumed": 2000,
            "distribution": {entity["id"]: 500 for entity in entities}
        }
        
        # Create the world instance
        real_world = World(
            world_id=1,
            grid_size=GRID_SIZE,
            num_agents=len(entities),
            tasks=tasks,
            resources=resources
        )
        
        # Summarize the world's state
        summary = real_world.summarize_state()
        add_log(f"Initialized and summarized real simulation world: {summary}")
        
        # Return the summary as a JSON response
        return JSONResponse({
            "status": "Real simulation world summarized successfully.",
            "summary": summary
        })
    except Exception as e:
        error_message = f"Error initializing and summarizing world: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)
    
# Endpoint to reset simulation
@router.post("/reset", tags=["World Simulation"])
async def reset_simulation():
    """
    Reset the simulation. Permanently deletes the proper keys, and initializes the simulation with entity data, and randomizes positions.
    """
    global stop_signal
    try:
        add_log("Reset simulation process initiated.")
        stop_signal = False
        await redis.flushdb()
        add_log("Redis database flushed successfully.")
        entities = await initialize_entities()
        add_log(f"Entities reinitialized successfully. Total entities: {len(entities)}")
        return JSONResponse({"status": "Simulation reset successfully.", "entities": entities})
    except Exception as e:
        error_message = f"Error during simulation reset: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

# Endpoint to initialize simulation
@router.post("/initialize", tags=["World Simulation"])
async def initialize_simulation():
    """
    Initialize a new world by setting the entities positions at random and assigning names and ids.
    """
    global stop_signal
    try:
        add_log("Simulation initialization process started.")
        stop_signal = False
        entities = await initialize_entities()
        add_log(f"Entities initialized successfully. Total entities: {len(entities)}")
        return JSONResponse({"status": "Simulation started successfully.", "entities": entities})
    except Exception as e:
        error_message = f"Error during simulation initialization: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@router.post("/stop", tags=["World Simulation"])
async def stop_simulation():
    """
    Sends an iteration stop signal to prevent further iterations. After the current iteration is over it will not start a new one.
    """
    global stop_signal
    try:
        # Log the start of the stop process
        add_log("Stop simulation process initiated.")
        
        # Set the stop signal
        stop_signal = True
        add_log("Stop signal triggered successfully.")
        
        # Log successful completion
        add_log("Simulation stopping process completed.")
        return JSONResponse({"status": "Simulation stopping after completing the current iteration."})
    except Exception as e:
        # Log and raise any unexpected errors
        error_message = f"Error during simulation stop process: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)
