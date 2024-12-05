from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from config import (
    GRID_SIZE,
    NUM_ENTITIES,
    logger,
    add_log,
    REDIS_ENDPOINT,
    REDIS_PASSWORD,
)
from redis.asyncio import Redis
import random
import asyncio

# Initialize Router
router = APIRouter()

# Redis client
redis = Redis(
    host=REDIS_ENDPOINT,
    port=6379,
    password=REDIS_PASSWORD,
    decode_responses=True,
    ssl=True
)

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
    Send a message to an entity. 
    DIRECTIONS: Enter the integer of the entity you wish to message and execute. 
    The response body reveals any entities within the messaging range of the entity you wish to message. 
    Note the "id" of the entity and then use "Send Messsage" to send a message as the "id" of that entity so your chosen entity recieves the message. 
    The sending entity will not have a memory of the message being sent.
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
    Send a message to an entity. 
    DIRECTIONS: Enter the integer of the entity you wish to message and execute. 
    The response body reveals any entities within the messaging range of the entity you wish to message. 
    Note the "id" of the entity and then use "Send Messsage" to send a message as the "id" of that entity so your chosen entity recieves the message. 
    The sending entity will not have a memory of the message being sent.
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
