import json
import logging
import os
import psycopg2
import redis
import time
import tracemalloc
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from endpoints.database import redis, supabase
from utils import add_log, LOG_QUEUE, logger
from config import NUM_ENTITIES, LOG_FILE, LOG_DIR
from models import Entity
from collections import deque
from logging.handlers import RotatingFileHandler
from supabase import Client, create_client

router = APIRouter()


# Global logger instance
logger = logging.getLogger("simulation_app")

# Global variables for maxBytes and backupCount
max_bytes = 10**6  # 1 MB by default
backup_count = 3  # Keep 3 backups by default

# Global deque for logs
LOG_QUEUE = deque(maxlen=100)  # Default maxlen is 100

log_file_path = os.path.join(os.getcwd(), 'simulation_logs.log')

def setup_rotating_handler():
    """Setup rotating handler with the current maxBytes and backupCount."""
    return RotatingFileHandler(
        'simulation_logs.log', mode='a', maxBytes=max_bytes, backupCount=backup_count
    )

# Endpoint to retrieve aggregated statistics about the simulation (e.g., entities, messages, memory usage)
@router.get("/analytics", tags=["Utilities"])
async def analytics_dashboard():
    """
    Retrieve aggregated statistics about the simulation.
    """
    try:
        entity_keys = await redis.keys("entity:*")
        total_entities = len(entity_keys)
        total_messages = 0
        memory_sizes = []

        for key in entity_keys:
            entity_data = await redis.hgetall(key)
            messages = await redis.lrange(f"{entity_data['id']}:messages", 0, -1)
            total_messages += len(messages)
            memory_sizes.append(len(entity_data.get("memory", "")))

        avg_memory_size = sum(memory_sizes) / len(memory_sizes) if memory_sizes else 0
        avg_messages = total_messages / total_entities if total_entities > 0 else 0

        analytics = {
            "total_entities": total_entities,
            "total_messages": total_messages,
            "average_memory_size": avg_memory_size,
            "average_messages_per_entity": avg_messages,
        }

        add_log("Analytics dashboard data generated successfully.")
        return analytics
    except Exception as e:
        error_message = f"Error generating analytics dashboard: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)


# Endpoint to download the current state of the simulation as a JSON file
@router.get("/simulation/state/download", tags=["Utilities"])
async def download_simulation_state():
    """
    Download the current state of the simulation as a JSON file.
    """
    try:
        # Collect all entities from Redis
        entity_keys = await redis.keys("entity:*")
        entities = []
        for key in entity_keys:
            entity_data = await redis.hgetall(key)
            messages = await redis.lrange(f"{entity_data['id']}:messages", 0, -1)
            entity_data["messages"] = messages
            entities.append(entity_data)

        # Write to a temporary JSON file
        file_path = "/tmp/simulation_state.json"
        with open(file_path, "w") as f:
            json.dump(entities, f)

        add_log("Simulation state saved to JSON file.")
        return FileResponse(file_path, media_type="application/json", filename="simulation_state.json")
    except Exception as e:
        error_message = f"Error downloading simulation state: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

# Endpoint to upload and restore a saved simulation state from a JSON file
@router.post("/simulation/state/upload", tags=["Utilities"])
async def upload_simulation_state(state_file: UploadFile = File(...)):
    """
    Upload and restore a saved simulation state.
    """
    try:
        # Read the uploaded file content
        file_content = await state_file.read()

        # Attempt to decode the file content as JSON
        try:
            state_data = json.loads(file_content)
        except json.JSONDecodeError as e:
            error_message = f"Invalid JSON format: {str(e)}"
            add_log(error_message)
            raise HTTPException(status_code=400, detail=error_message)

        # Clear current Redis data
        await redis.flushdb()
        add_log("Redis database cleared before state restoration.")

        # Restore entities and messages
        for entity in state_data:
            entity_key = f"entity:{entity['id']}"
            messages_key = f"{entity['id']}:messages"
            messages = entity.pop("messages", [])
            await redis.hset(entity_key, mapping=entity)
            for message in messages:
                await redis.lpush(messages_key, message)

        add_log("Simulation state restored successfully.")
        return {"status": "Simulation state uploaded and restored."}
    
    except Exception as e:
        # General error handler for other issues
        error_message = f"Error uploading simulation state: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)
    
# Endpoint to synchronize all entities between Redis and Supabase
@router.post("/sync_entity", tags=["Utilities"])
async def sync_entity():
    """
    Synchronize all entities between Redis and Supabase.
    """
    try:
        add_log("Synchronization process initiated between Redis and Supabase.")

        all_entities = [
            Entity(**await redis.hgetall(f"entity:{i}"))
            for i in range(NUM_ENTITIES)
        ]

        for entity in all_entities:
            supabase.table("entities").upsert(entity.dict()).execute()
            add_log(f"Entity with ID {entity.id} synchronized to Supabase.")

        add_log("Synchronization process completed successfully.")
        return {"status": "Entities synchronized between Redis and Supabase"}
    except Exception as e:
        error_message = f"Error during entity synchronization: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

# Endpoint to test and report the status of Redis connectivity
@router.get("/debug/redis", tags=["Utilities"])
async def test_redis_connectivity():
    """
    Test and report the status of Redis connectivity.
    """
    try:
        # Check if Redis is reachable by pinging it
        ping_response = await redis.ping()
        if ping_response:
            add_log("Redis connectivity test successful.")
            return {"status": "connected", "message": "Redis is reachable."}
        else:
            # Redis returned a falsy value
            raise Exception("Redis ping returned a falsy value.")
    except redis.exceptions.ConnectionError as e:
        # Handle Redis connection errors (e.g., wrong host, port)
        error_message = f"Redis connectivity test failed: Connection error - {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)
    except redis.exceptions.TimeoutError as e:
        # Handle Redis timeout errors
        error_message = f"Redis connectivity test failed: Timeout error - {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)
    except Exception as e:
        # General error handler for any other issues
        error_message = f"Redis connectivity test failed: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)


# Endpoint to test and report the status of Supabase connectivity
@router.get("/debug/supabase", tags=["Utilities"])
async def test_supabase_connectivity():
    """
    Test and report the status of Supabase connectivity.
    """
    try:
        # Query Supabase to check if the database is reachable
        response = supabase.table("entities").select("*").limit(1).execute()

        if response.data:
            add_log("Supabase connectivity test successful.")
            return {"status": "connected", "message": "Supabase is reachable."}
        else:
            # Handle case where the connection is successful, but no data is returned
            raise Exception("Supabase query returned no data, though connection was successful.")
    except psycopg2.OperationalError as e:
        # Handle database connection errors (e.g., wrong credentials, host, etc.)
        error_message = f"Supabase connectivity test failed: Database connection error - {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)
    except Exception as e:
        # General error handler for any other issues
        error_message = f"Supabase connectivity test failed: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)
    
# Endpoint to profile the simulation and identify performance bottlenecks
@router.get("/debug/performance", tags=["Utilities"])
async def profile_simulation_performance():
    """
    Profile the simulation to identify bottlenecks in processing or memory usage.
    """
    try:
        tracemalloc.start()
        start_time = time.time()

        # Simulate fetching all entities
        entity_keys = await redis.keys("entity:*")
        add_log(f"Starting performance profiling. Number of entities to process: {len(entity_keys)}")
        
        # Log entity keys for better traceability
        # Decode keys if they are bytes
        entity_keys_decoded = [key.decode() if isinstance(key, bytes) else key for key in entity_keys]
        add_log(f"Entity keys being processed: {', '.join(entity_keys_decoded)}")

        for key in entity_keys:
            await redis.hgetall(key)

        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        performance_data = {
            "execution_time": f"{end_time - start_time:.2f} seconds",
            "memory_usage": f"{current / 10**6:.2f} MB",
            "peak_memory_usage": f"{peak / 10**6:.2f} MB"
        }

        # Log the results
        add_log(f"Performance profiling completed: {performance_data}")

        return JSONResponse(performance_data)
    except Exception as e:
        error_message = f"Error during performance profiling: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)
      
# Endpoint to inspect a specific Redis key (e.g., entity:0) for debugging purposes
@router.get("/debug/redis/{key}", tags=["Utilities"])
async def inspect_redis_key(key: str):
    """
    Fetch the content of a specific Redis key for debugging purposes. ie entity:0
    """
    try:
        key_type = await redis.type(key)
        if key_type == "none":
            error_message = f"Redis key '{key}' does not exist."
            add_log(error_message)
            raise HTTPException(status_code=404, detail=error_message)

        if key_type == "string":
            value = await redis.get(key)
        elif key_type == "list":
            value = await redis.lrange(key, 0, -1)
        elif key_type == "hash":
            value = await redis.hgetall(key)
        else:
            value = f"Unsupported key type: {key_type}"

        add_log(f"Inspected Redis key '{key}': {value}")
        return {"key": key, "type": key_type, "value": value}
    except Exception as e:
        error_message = f"Error inspecting Redis key '{key}': {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)
