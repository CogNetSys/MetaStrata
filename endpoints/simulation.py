# /endpoints/simulation.py (renamed from simulation_control.py for clarity)

import asyncio
from typing import List, Optional, Dict
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import json
import random
import logging

from core import app
from endpoints.database import redis
from endpoints.mtnn import submit_summary
from utils import add_log, logger
from world import World  # Ensure World class is correctly imported
from config import GRID_SIZE, NUM_ENTITIES, NUM_WORLDS

# Initialize Router
router = APIRouter()

# Stop signal for simulation control
stop_signal = False

# Pydantic Model for Specifying Worlds
class WorldSelection(BaseModel):
    world_ids: Optional[List[int]] = Field(
        None,
        description="List of world IDs to apply the action. If not provided, action applies to all worlds."
    )

# Helper Function to Reset a Specific World
async def reset_world(world_id: int):
    """
    Reset a specific world by clearing its data and re-initializing it.
    """
    try:
        tasks_key = f"world:{world_id}:tasks"
        resources_key = f"world:{world_id}:resources"

        # Ensure the tasks key exists
        if not await redis.exists(tasks_key):
            add_log(f"Tasks key for World {world_id} not found. Initializing empty tasks.")
            await redis.hset(tasks_key, "_init", "placeholder")  # Create empty tasks key

        # Ensure the resources key exists
        if not await redis.exists(resources_key):
            add_log(f"Resources key for World {world_id} not found. Initializing empty resources.")
            await redis.hset(resources_key, "_init", "placeholder")  # Create empty resources key

        # Clear and reinitialize data
        await redis.delete(f"world:{world_id}:messages")
        await redis.hset(f"world:{world_id}:message", "message", "")  # Reset message field

        # Reinitialize tasks and resources as per your configuration
        tasks = [
            {
                "id": t,
                "progress": 0.0,
                "duration": random.randint(5, 20),
                "increment": 0.1,
                "priority": 1.0,
            }
            for t in range(100)
        ]

        resources = {
            "total": 1000,
            "consumed": 0,
            "distribution": {agent_id: 100 for agent_id in range(NUM_ENTITIES)},
        }

        await redis.hmset(tasks_key, {f"task_{task['id']}": json.dumps(task) for task in tasks})
        await redis.hmset(resources_key, {
            "total": resources["total"],
            "consumed": resources["consumed"],
            "distribution": json.dumps(resources["distribution"])
        })

        add_log(f"World {world_id} has been reset successfully.")
    except Exception as e:
        logger.error(f"Error resetting World {world_id}: {e}")
        add_log(f"Error resetting World {world_id}: {e}")
        raise

# Helper Function to Initialize a Specific World
async def initialize_world(world_id: int):
    """
    Initialize a specific world in the simulation with pipelining for Redis.

    This function creates all necessary Redis keys for a world, including:
    - Stream for messaging
    - Tasks
    - Resources
    - Entities
    - Stop signal and messages
    """
    try:
        # Define all necessary keys
        stream_key = f"world:{world_id}:stream"
        tasks_key = f"world:{world_id}:tasks"
        resources_key = f"world:{world_id}:resources"
        stop_signal_key = f"world:{world_id}:stop_signal"
        message_key = f"world:{world_id}:message"
        entity_prefix = f"world:{world_id}:entity:"
        
        # Debug log to confirm function is called
        add_log(f"Initializing World {world_id}.")

        # Check if the world already exists by checking 'world:{world_id}:tasks'
        if await redis.exists(tasks_key):
            add_log(f"World {world_id} already exists. Skipping initialization.")
            return

        # Create tasks and resources
        tasks = [
            {
                "id": t,
                "progress": 0.0,
                "duration": random.randint(5, 20),
                "increment": 0.1,
                "priority": 1.0,
            }
            for t in range(100)
        ]

        resources = {
            "total": 1000,
            "consumed": 0,
            "distribution": {agent_id: 100 for agent_id in range(NUM_ENTITIES)},
        }

        # Prepare the pipelined commands
        pipeline_commands = []

        # Initialize Streams with a dummy entry to ensure stream exists
        pipeline_commands.append(["XADD", stream_key, "*", "init", "stream_initialized"])

        # Create Consumer Group for the stream
        try:
            pipeline_commands.append(["XGROUP", "CREATE", stream_key, "group_1", "0", "MKSTREAM"])
            add_log(f"Consumer group 'group_1' created for World {world_id}.")
        except Exception as e:
            # If the group already exists, log and continue
            if "BUSYGROUP" in str(e).upper():
                add_log(f"Consumer group 'group_1' already exists for World {world_id}.")
            else:
                raise

        # Set stop_signal and message
        pipeline_commands.extend([
            ["SET", stop_signal_key, "False"],
            ["HSET", message_key, "message", ""]
        ])

        # Add tasks to the pipeline using HSET with mapping
        tasks_mapping = {f"task_{task['id']}": json.dumps(task) for task in tasks}
        pipeline_commands.append(["HSET", tasks_key, *[item for pair in tasks_mapping.items() for item in pair]])

        # Add resources to the pipeline using HSET with mapping
        resources_mapping = {
            "total": resources["total"],
            "consumed": resources["consumed"],
            "distribution": json.dumps(resources["distribution"])
        }
        pipeline_commands.append(["HSET", resources_key, *[item for pair in resources_mapping.items() for item in pair]])

        # Initialize entities with proper namespacing
        for entity_id in range(NUM_ENTITIES):
            entity_key = f"{entity_prefix}{entity_id}"
            entity_data = {
                "id": entity_id,
                "x": random.randint(0, GRID_SIZE - 1),  # Assuming GRID_SIZE=30
                "y": random.randint(0, GRID_SIZE - 1),
                "memory": "",
                "cooperation_score": random.random(),
                "conflict_score": random.random(),
            }
            # Use HSET with mapping for each entity
            pipeline_commands.append(["HSET", entity_key, "data", json.dumps(entity_data)])

        # Send the pipeline commands to Redis
        async with redis.pipeline(transaction=False) as pipeline:
            for command in pipeline_commands:
                pipeline.execute_command(*command)
            await pipeline.execute()

        # Verify the stream exists after initialization
        if not await redis.exists(stream_key):
            add_log(f"Stream {stream_key} was not created. Retrying.")
            raise RuntimeError(f"Stream {stream_key} was not successfully created for World {world_id}.")

        add_log(f"World {world_id} has been initialized successfully with pipelining.")

    except Exception as e:
        logger.error(f"Error initializing World {world_id}: {e}")
        add_log(f"Error initializing World {world_id}: {e}")
        raise

@router.get("/active_worlds", tags=["Simulation"])
async def get_active_worlds():
    """
    Retrieve a list of all unique active world IDs.

    ### Response Format
    - `active_worlds`: List of integers representing the unique IDs of active worlds.

    #### Example Response
    ```json
    {
      "active_worlds": [0, 1, 2, 3]
    }
    ```
    """
    try:
        # Fetch all keys matching the "world:*:tasks" pattern to identify active worlds
        world_tasks_keys = await redis.keys("world:*:tasks")
        active_worlds = sorted(
            {int(key.split(":")[1]) for key in world_tasks_keys}
        )

        add_log(f"Fetched active worlds: {active_worlds}")
        return {"active_worlds": active_worlds}
    except Exception as e:
        error_message = f"Error fetching active worlds: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=f"Custom Error: {error_message}")
    
@router.post("/cleanup_and_initialize", tags=["Simulation"])
async def cleanup_and_initialize():
    """
    **Developer-Only Endpoint**

    Cleans up all existing Redis keys and reinitializes all worlds based on `NUM_WORLDS`.
    **Use with caution!**

    ### Example Request
    ```json
    {}
    ```

    ### Example Response
    ```json
    {
      "status": "Cleanup and initialization completed.",
      "worlds_initialized": [0, 1, 2, 3],
      "worlds_failed": {}
    }
    ```
    """
    try:
        # Flush all Redis data
        await redis.flushdb()
        add_log("Redis database has been flushed.")

        # Initialize all worlds
        success = []
        failures = {}
        for world_id in range(0, NUM_WORLDS + 1):
            try:
                await initialize_world(world_id)
                await reset_world(world_id)
                success.append(world_id)
            except Exception as e:
                failures[world_id] = str(e)

        response = {
            "status": "Cleanup and initialization completed.",
            "worlds_initialized": success,
            "worlds_failed": failures
        }

        return JSONResponse(response)
    except Exception as e:
        error_message = f"Error during cleanup and initialization: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=f"Custom Error: {error_message}")
    
# Endpoint to Initialize and Reset Worlds
@router.post("/initialize_and_reset", tags=["Simulation"])
async def initialize_and_reset_simulation(selection: WorldSelection):
    """ 
    Initialize and reset the simulation for specified worlds.

    If no world IDs are specified in the request body, the endpoint will reset **all worlds**. 
    This operation reinitializes the worlds' tasks, resources, and messages.

    ### Request Body
    - `world_ids` (optional): A list of integers representing the IDs of the worlds to reset. If omitted or an empty list, all worlds are reset.

    ### Examples
    #### Reset a Specific World
    ```json
    {
      "world_ids": [0]
    }
    ```

    #### Reset Multiple Worlds
    ```json
    {
      "world_ids": [1, 2, 3]
    }
    ```

    #### Reset All Worlds (Default)
    If no `world_ids` are provided, all worlds will be reset:
    ```json
    {}
    ```

    Alternatively, explicitly specify an empty list to reset all worlds:
    ```json
    {
      "world_ids": []
    }
    ```

    #### Reset a Range of Worlds
    To reset worlds 4 through 10:
    ```json
    {
      "world_ids": [4, 5, 6, 7, 8, 9, 10]
    }
    ```

    ### Response Format
    - `status`: A message summarizing the action.
    - `worlds_processed`: List of worlds successfully reset.
    - `worlds_failed`: Dictionary of worlds that failed to reset, along with error details.

    #### Example Response
    ```json
    {
      "status": "Simulation initialize and reset process completed.",
      "worlds_processed": [1, 2, 3],
      "worlds_failed": {
        "4": "404: World with ID 4 does not exist.",
        "5": "404: World with ID 5 does not exist."
      }
    }
    ```
    """
    try:
        if selection.world_ids:
            target_worlds = selection.world_ids
            add_log(f"Initialize and reset initiated for worlds: {target_worlds}")
        else:
            # Default to all worlds up to NUM_WORLDS
            target_worlds = list(range(0, NUM_WORLDS + 1))  # Including world 0
            add_log("Initialize and reset initiated for all worlds.")

        # Filter only existing worlds
        existing_worlds = []
        failures = {}
        for world_id in target_worlds:
            exists = await redis.exists(f"world:{world_id}:tasks")
            if exists:
                existing_worlds.append(world_id)
            else:
                failures[world_id] = "World does not exist."

        if not existing_worlds and not failures:
            add_log("No worlds found to initialize and reset.")
            return JSONResponse({
                "status": "No worlds found to initialize and reset.",
                "worlds_processed": [],
                "worlds_failed": failures
            })

        success = []
        # Reset existing worlds
        for world_id in existing_worlds:
            try:
                await initialize_world(world_id)
                await reset_world(world_id)
                success.append(world_id)
            except HTTPException as he:
                failures[world_id] = f"{he.status_code}: {he.detail}"
            except Exception as e:
                failures[world_id] = str(e)

        response = {
            "status": "Simulation initialize and reset process completed.",
            "worlds_processed": success,
            "worlds_failed": failures
        }

        return JSONResponse(response)
    except Exception as e:
        error_message = f"Error during initialize and reset: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=f"Custom Error: {error_message}")

# Endpoint to Stop Simulation
@router.post("/stop", tags=["Simulation"])
async def stop_simulation(selection: WorldSelection):
    """
    Stop the simulation for specified worlds. If no worlds are specified, stop all worlds.
    """
    global stop_signal
    try:
        # Determine target worlds
        if selection.world_ids:
            target_worlds = selection.world_ids
            add_log(f"Stop simulation initiated for worlds: {target_worlds}")
        else:
            # Fetch all world IDs (excluding keys with ":messages")
            world_keys = await redis.keys("world:*:tasks")
            target_worlds = [int(key.split(":")[1]) for key in world_keys if ":messages" not in key]
            add_log("Stop simulation initiated for all worlds.")

        if not target_worlds:
            add_log("No worlds found to stop.")
            raise HTTPException(status_code=404, detail="No worlds found to stop.")

        success = []
        failures = {}

        # Stop each specified world
        for world_id in target_worlds:
            try:
                await stop_world(world_id)
                success.append(world_id)
            except HTTPException as he:
                failures[world_id] = f"{he.status_code}: {he.detail}"
            except Exception as e:
                failures[world_id] = str(e)

        response = {
            "status": "Simulation stop process completed.",
            "worlds_stopped": success,
            "worlds_failed": failures
        }

        return JSONResponse(response)
    except HTTPException as he:
        raise he
    except Exception as e:
        error_message = f"Error during simulation stop: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=f"Custom Error: {error_message}")
    
# @router.post("/worlds/{world_id}/step", tags=["Simulation"])
# async def perform_steps_for_world(world_id: int, request: StepRequest):
#     global stop_signal
#     stop_signal = False  # Reset stop signal before starting steps

#     try:
#         for step in range(request.steps):
#             if stop_signal:
#                 add_log("Simulation steps halted by stop signal.")
#                 break

#             # Perform simulation logic for the specific world
#             world = next((w for w in app.state.worlds if w.world_id == world_id), None)
#             if world:
#                 await simulate_world_step(world)
#                 summary_vector = world.summarize_state()
#                 await submit_summary(world.world_id, summary_vector)
#                 add_log(f"World {world_id} summary: {summary_vector}")
#             else:
#                 raise HTTPException(status_code=404, detail=f"World {world_id} not found.")

#         add_log(f"Simulation steps completed for World {world_id}: {request.steps} step(s).")
#         return {"status": f"Performed {request.steps} step(s) for World {world_id}."}
#     except Exception as e:
#         logger.error(f"Error during simulation for World {world_id}: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error during simulation for World {world_id}: {str(e)}")

