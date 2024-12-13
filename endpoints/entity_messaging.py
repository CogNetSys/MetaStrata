# /endpoints/entity_messaging.py

import asyncio
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from endpoints.database import redis
from utils import add_log, logger
from config import NUM_ENTITIES, CHEBYSHEV_DISTANCE

from pydantic import BaseModel

# Pydantic Models
class Entity(BaseModel):
    id: int
    name: str
    x: int
    y: int
    memory: Optional[str] = ""

class BatchMessage(BaseModel):
    entity_id: int
    message: str

class BatchMessagesPayload(BaseModel):
    messages: List[BatchMessage]

# Helper function for Chebyshev distance
def chebyshev_distance(x1, y1, x2, y2):
    return max(abs(x1 - x2), abs(y1 - y2))

# Initialize APIRouter for Entity Messaging
router = APIRouter(
    prefix="/entities",
    tags=["Entity Messaging"]
)

@router.get("/{entity_id}/messages", response_model=Optional[str])
async def retrieve_message(entity_id: int):
    """
    Retrieve the current message for a specific entity.
    """
    try:
        # Validate that the entity exists
        entity_key = f"entity:{entity_id}"
        if not await redis.exists(entity_key):
            error_message = f"Entity with ID {entity_id} not found."
            add_log(error_message)
            raise HTTPException(status_code=404, detail=error_message)

        # Fetch the message field for the entity
        message = await redis.hget(entity_key, "message")
        if not message:
            add_log(f"No current message found for Entity {entity_id}.")
            return None  # Return None if no message exists

        add_log(f"Retrieved current message for Entity {entity_id}: \"{message}\".")
        return message
    except Exception as e:
        error_message = f"Error retrieving message for Entity {entity_id}: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@router.post("/{recipient_id}/create_memory", tags=["Entity Messaging"])
async def create_memory(
    recipient_id: int,
    message: str = Query(..., description="The memory content to add or update for the recipient entity.")
):
    """
    Create a memory for an entity.
    DIRECTIONS: Append a memory to their existing memory field.
    """
    recipient_key = f"entity:{recipient_id}"

    try:
        # Log the attempt to create memory
        add_log(f"Creating a memory for Entity {recipient_id}: \"{message}\".")

        # Validate that the recipient exists
        if not await redis.exists(recipient_key):
            error_message = f"Recipient Entity ID {recipient_id} not found."
            add_log(error_message)
            raise HTTPException(status_code=404, detail=error_message)

        # Fetch the recipient's existing memory field
        existing_memory = await redis.hget(recipient_key, "memory")
        if existing_memory:
            # Append the new message to the recipient's memory
            updated_memory = f"{existing_memory}\n{message}"
        else:
            # Start the memory with the new message
            updated_memory = message

        # Update the recipient's memory field
        await redis.hset(recipient_key, "memory", updated_memory)
        add_log(f"Memory updated successfully for Entity {recipient_id}: \"{message}\".")

        return {"status": "Memory updated successfully", "message": message}

    except Exception as e:
        # Log and raise any unexpected errors
        error_message = f"Error creating memory for Entity {recipient_id}: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@router.post("/{recipient_id}/send_message", tags=["Entity Messaging"])
async def send_message(
    recipient_id: int,
    message: str = Query(..., description="The message to send to the recipient entity.")
):
    """
    Send a message as an entity.
    DIRECTIONS: Append a message to a designated entity's existing message field.
    This means you are sending a message as the designated entity to the surrounding entities.
    The designated entity does not receive the message.
    """
    recipient_key = f"entity:{recipient_id}"

    try:
        # Log the attempt to send a message
        add_log(f"Attempting to send a message to Entity {recipient_id}: \"{message}\".")

        # Validate that the recipient exists
        if not await redis.exists(recipient_key):
            error_message = f"Recipient Entity ID {recipient_id} not found."
            add_log(error_message)
            raise HTTPException(status_code=404, detail=error_message)

        # Fetch the existing message (if any)
        existing_message = await redis.hget(recipient_key, "message") or ""

        # Append the new message, separating with a delimiter if needed
        updated_message = existing_message + "\n" + message if existing_message else message

        # Update the `message` field with the appended message
        await redis.hset(recipient_key, "message", updated_message)

        # Log the successful message update
        add_log(f"Message appended successfully to Entity {recipient_id}: \"{message}\".")

        return {"status": "Message appended successfully", "recipient_id": recipient_id, "message": updated_message}

    except Exception as e:
        # Log and raise any unexpected errors
        error_message = f"Error sending message to Entity {recipient_id}: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@router.get("/{entity_id}/nearby", response_model=List[Entity], tags=["Entity Messaging"])
async def get_nearby_entities(entity_id: int):
    """
    Get entities within the messaging range of a specific entity.
    DIRECTIONS: Enter the integer of the entity you wish to message and execute.
    The response body reveals any entities within the messaging range of the entity you wish to message.
    Note the "id" of the entity and then use "Send Message" to send a message as the "id" of that entity so your chosen entity receives the message.
    The sending entity will not have a memory of the message being sent.
    """
    try:
        # Log the attempt to fetch nearby entities
        add_log(f"Fetching nearby entities for Entity ID {entity_id}.")

        # Get the entity's position from Redis
        entity_data = await redis.hgetall(f"entity:{entity_id}")
        if not entity_data:
            error_message = f"Entity with ID {entity_id} not found."
            add_log(error_message)
            raise HTTPException(status_code=404, detail=error_message)

        entity = Entity(**entity_data)

        # Fetch all entities except the current one
        all_entities_data = await asyncio.gather(*[
            redis.hgetall(f"entity:{i}") for i in range(NUM_ENTITIES) if i != entity_id
        ])
        all_entities = []
        for data in all_entities_data:
            if data:
                all_entities.append(Entity(**data))

        # Filter nearby entities based on Chebyshev distance
        nearby_entities = [
            a for a in all_entities
            if chebyshev_distance(entity.x, entity.y, a.x, a.y) <= CHEBYSHEV_DISTANCE
        ]

        add_log(f"Nearby entities fetched successfully for Entity ID {entity_id}. Total nearby entities: {len(nearby_entities)}.")
        return nearby_entities
    except Exception as e:
        # Log and raise any unexpected errors
        error_message = f"Error fetching nearby entities for Entity ID {entity_id}: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@router.post("/update_batch_memory", tags=["Entity Messaging"])
async def update_batch_memory(payload: BatchMessagesPayload):
    """
    Update the `memory` field of multiple entities in one request.

    This function appends the provided content to the `memory` field of the respective entities in Redis.
    """
    try:
        updated_entities = []
        for msg in payload.messages:
            entity_key = f"entity:{msg.entity_id}"
            
            # Validate that the entity exists
            if not await redis.exists(entity_key):
                add_log(f"Entity {msg.entity_id} not found. Skipping update.")
                continue
            
            # Fetch the current memory for the entity (if it exists)
            existing_memory = await redis.hget(entity_key, "memory") or ""
            
            # Append the new content to the existing memory, separated by a newline
            updated_memory = f"{existing_memory}\n{msg.message}".strip()
            
            # Update the `memory` field with the appended content
            await redis.hset(entity_key, "memory", updated_memory)
            updated_entities.append(msg.entity_id)
        
        add_log(f"Batch memory updates applied to Entities: {updated_entities}.")
        return {"status": "success", "updated_entities": updated_entities}
    
    except Exception as e:
        error_message = f"Custom Error: Error updating batch memory: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@router.delete("/{entity_id}/memory", tags=["Entity Messaging"])
async def clear_memory(entity_id: int):
    """
    Wipe an entity's memory field.
    """
    try:
        entity_key = f"entity:{entity_id}"
        
        # Validate entity exists
        if not await redis.exists(entity_key):
            error_message = f"Entity with ID {entity_id} not found."
            add_log(error_message)
            raise HTTPException(status_code=404, detail=error_message)
        
        # Clear the memory field
        await redis.hset(entity_key, "memory", "")
        add_log(f"Memory cleared for Entity {entity_id}.")
        return {"status": "success", "message": f"Memory cleared for Entity {entity_id}."}
    except Exception as e:
        error_message = f"Error clearing memory for Entity {entity_id}: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@router.post("/broadcast_message", tags=["Entity Messaging"])
async def broadcast_message(message: str):
    """
    Broadcast a message to all entities.
    """
    try:
        # Fetch all entity keys
        entity_keys = await redis.keys("entity:*")
        if not entity_keys:
            add_log("No entities found to broadcast the message.")
            raise HTTPException(status_code=404, detail="No entities found.")

        # Broadcast the message to all entities
        for key in entity_keys:
            entity_id = key.decode().split(":")[1]  # Extract entity ID
            message_key = f"entity:{entity_id}:messages"
            await redis.lpush(message_key, message)

        # Log the broadcast action
        add_log(f"Broadcast message to all entities: \"{message}\".")
        return {"status": "success", "message": f"Broadcasted message to {len(entity_keys)} entities."}
    except Exception as e:
        error_message = f"Error broadcasting message: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@router.delete("/{entity_id}/messages", tags=["Entity Messaging"])
async def clear_all_messages(entity_id: int):
    """
    Remove all messages for a specific entity.
    """
    try:
        message_key = f"entity:{entity_id}:messages"

        # Validate that the message key exists
        if not await redis.exists(message_key):
            error_message = f"No messages found for Entity {entity_id}."
            add_log(error_message)
            raise HTTPException(status_code=404, detail=error_message)

        # Clear all messages
        await redis.delete(message_key)
        add_log(f"All messages cleared for Entity {entity_id}.")
        return {"status": "success", "message": f"All messages cleared for Entity {entity_id}."}
    except Exception as e:
        error_message = f"Error clearing messages for Entity {entity_id}: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)
