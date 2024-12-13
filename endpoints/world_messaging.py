# /endpoints/world_messaging.py

from typing import Optional, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from endpoints.database import redis
from utils import add_log, logger
from config import NUM_WORLDS

import json
import httpx

# Pydantic Models
class WorldMessage(BaseModel):
    source_world_id: int
    message_type: str
    payload: Dict

# Initialize APIRouter for Worlds Messaging
router = APIRouter(
    tags=["World Messaging"]
)

@router.post("/{world_id}/send_message", tags=["World Messaging"])
async def send_world_message(
    world_id: int,
    message: WorldMessage
):
    """
    Send a directed message to a specific world using Redis Streams.
    """
    try:
        # Validate that the world exists by checking 'world:{id}:tasks'
        world_tasks_key = f"world:{world_id}:tasks"
        if not await redis.exists(world_tasks_key):
            error_message = f"World with ID {world_id} not found."
            add_log(error_message)
            raise HTTPException(status_code=404, detail=error_message)

        # Prepare the message
        world_message = {
            "source_world_id": str(message.source_world_id),
            "message_type": message.message_type,
            "payload": json.dumps(message.payload)
        }

        # Add the message to the stream
        stream_key = f"world:{world_id}:stream"
        await redis.xadd(stream_key, world_message)

        # Log the message sending
        add_log(f"Message sent to World {world_id}: \"{message.message_type}\" with payload {message.payload}.")

        return {"status": "success", "message": f"Message sent to World {world_id}."}

    except Exception as e:
        error_message = f"Error sending message to World {world_id}: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@router.post("/broadcast_message", tags=["World Messaging"])
async def broadcast_world_message(message_type: str, payload: Dict):
    """
    Broadcast a message to all connected worlds using Redis Streams.
    """
    try:
        # Fetch all world tasks keys to identify existing worlds
        world_tasks_keys = await redis.keys("world:*:tasks")
        if not world_tasks_keys:
            add_log("No worlds found to broadcast the message.")
            raise HTTPException(status_code=404, detail="No worlds found.")

        # Prepare the message
        world_message = {
            "source_world_id": "broadcast",
            "message_type": message_type,
            "payload": json.dumps(payload)
        }

        # Initialize Redis pipeline
        pipeline = redis.pipeline()
        for key in world_tasks_keys:
            world_id = key.decode().split(":")[1]  # Extract world ID
            stream_key = f"world:{world_id}:stream"
            pipeline.xadd(stream_key, world_message)

        # Execute pipeline
        await pipeline.execute()

        # Log the broadcast action
        add_log(f"Broadcast message to all worlds: \"{message_type}\" with payload {payload}.")
        return {"status": "success", "message": f"Broadcasted message to {len(world_tasks_keys)} worlds."}

    except Exception as e:
        error_message = f"Error broadcasting message to worlds: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@router.get("/{world_id}/messages", response_model=Optional[Dict], tags=["World Messaging"])
async def retrieve_world_message(world_id: int):
    """
    Retrieve the latest message for a specific world.
    Note: This reads the most recent message without marking it as read.
    """
    try:
        # Validate that the world exists by checking 'world:{id}:tasks'
        world_tasks_key = f"world:{world_id}:tasks"
        if not await redis.exists(world_tasks_key):
            error_message = f"World with ID {world_id} not found."
            add_log(error_message)
            raise HTTPException(status_code=404, detail=error_message)

        # Read the latest message from the stream
        stream_key = f"world:{world_id}:stream"
        messages = await redis.xrevrange(stream_key, count=1)
        if not messages:
            add_log(f"No current messages found for World {world_id}.")
            return None  # Return None if no message exists

        # Extract message details
        message_id, message_data = messages[0]
        message_type = message_data.get(b"message_type", b"").decode('utf-8')
        payload = json.loads(message_data.get(b"payload", b"{}").decode('utf-8'))
        source_world_id = message_data.get(b"source_world_id", b"").decode('utf-8')

        add_log(f"Retrieved current message for World {world_id}: \"{message_type}\" with payload {payload}.")
        return {"message_type": message_type, "payload": payload, "source_world_id": source_world_id}

    except Exception as e:
        error_message = f"Error retrieving message for World {world_id}: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)
