from fastapi import APIRouter, HTTPException
from endpoints.database import redis, supabase
from config import GRID_SIZE
from models import Entity
from utils import add_log, LOG_QUEUE, logger

router = APIRouter()

@router.get("/entities", tags=["Entities"])
async def list_all_entities():
    """
    Fetch details of all entities, including their current position, memory, and messages.
    """
    try:
        # Fetch all entity keys from Redis
        entity_keys = await redis.keys("entity:*")
        if not entity_keys:
            add_log("No entities found in Redis.")
            return []

        # Fetch and parse entity data, including messages
        entities = []
        for key in entity_keys:
            entity_data = await redis.hgetall(key)
            if not entity_data:
                continue

            # Fetch the current message for the entity
            entity_id = int(entity_data["id"])
            message = entity_data.get("message", "")

            # Append the entity with messages to the result
            entities.append({
                "id": entity_id,
                "name": entity_data.get("name"),
                "x": int(entity_data.get("x")),
                "y": int(entity_data.get("y")),
                "memory": entity_data.get("memory", ""),
                "messages": [message] if message else [],  # Wrap in a list for consistency
            })

        add_log(f"Fetched details for {len(entities)} entities, including messages.")
        return entities
    except Exception as e:
        error_message = f"Error fetching all entities: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

# Create new entity
@router.post("/entities", response_model=Entity, tags=["Entities"])
async def create_entity(entity: Entity):
    entity_key = f"entity:{entity.id}"

    try:
        # Log the attempt to create a new entity
        add_log(f"Attempting to create new entity with ID {entity.id} and name '{entity.name}'.")

        # Check if the ID already exists in Redis
        if await redis.exists(entity_key):
            error_message = f"Entity ID {entity.id} already exists in Redis."
            add_log(error_message)
            raise HTTPException(status_code=400, detail=error_message)

        # Check if the ID already exists in Supabase
        existing_entity = supabase.table("entities").select("id").eq("id", entity.id).execute()
        if existing_entity.data:
            error_message = f"Entity ID {entity.id} already exists in Supabase."
            add_log(error_message)
            raise HTTPException(status_code=400, detail=error_message)

        # Save entity data in Redis
        await redis.hset(entity_key, mapping=entity.dict())
        add_log(f"Entity data for ID {entity.id} saved in Redis.")

        # Add the entity key to the Redis list
        await redis.lpush("entity_keys", entity_key)
        add_log(f"Entity key for ID {entity.id} added to Redis entity_keys list.")

        # Save the entity in Supabase
        supabase.table("entities").insert(entity.dict()).execute()
        add_log(f"Entity data for ID {entity.id} saved in Supabase.")

        # Log the successful creation of the entity
        add_log(f"New entity created successfully: ID={entity.id}, Name={entity.name}, Position=({entity.x}, {entity.y})")
        
        return entity

    except Exception as e:
        # Log and raise unexpected errors
        error_message = f"Error creating entity with ID {entity.id}: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

# Get a specific entity
@router.get("/entities/{entity_id}", response_model=Entity, tags=["Entities"])
async def get_entity(entity_id: int):
    # Log the attempt to fetch entity data
    add_log(f"Fetching data for entity with ID {entity_id}.")

    # Fetch entity data from Redis
    entity_data = await redis.hgetall(f"entity:{entity_id}")
    if not entity_data:
        error_message = f"Entity with ID {entity_id} not found."
        add_log(error_message)
        raise HTTPException(status_code=404, detail=error_message)

    # Convert Redis data into an Entity model (ensure proper types are cast)
    try:
        entity = Entity(
            id=entity_id,
            name=entity_data.get("name"),
            x=int(entity_data.get("x")),
            y=int(entity_data.get("y")),
            memory=entity_data.get("memory", "")
        )
        add_log(f"Successfully fetched entity with ID {entity_id}, Name: {entity.name}, Position: ({entity.x}, {entity.y}).")
        return entity
    except Exception as e:
        error_message = f"Failed to parse data for entity ID {entity_id}: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

# Update an entity
@router.put("/entities/{entity_id}", response_model=Entity, tags=["Entities"])
async def update_entity(entity_id: int, entity: Entity):
    try:
        # Log the attempt to update entity data
        add_log(f"Attempting to update entity with ID {entity_id}.")

        # Update entity data in Redis
        await redis.hset(f"entity:{entity_id}", mapping=entity.dict())
        add_log(f"Entity with ID {entity_id} updated in Redis.")

        # Update entity in Supabase
        supabase.table("entities").update(entity.dict()).eq("id", entity_id).execute()
        add_log(f"Entity with ID {entity_id} updated in Supabase.")

        # Log successful update
        add_log(f"Successfully updated entity with ID {entity_id}.")
        return entity
    except Exception as e:
        # Log and raise error if update fails
        error_message = f"Error updating entity with ID {entity_id}: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

# Delete an entity
@router.delete("/entities/{entity_id}", tags=["Entities"])
async def delete_entity(entity_id: int):
    entity_key = f"entity:{entity_id}"
    try:
        # Log the attempt to delete an entity
        add_log(f"Attempting to delete entity with ID {entity_id}.")

        # Delete entity from Redis
        await redis.delete(entity_key)
        add_log(f"Entity with ID {entity_id} deleted from Redis.")

        # Remove the key from the Redis list
        await redis.lrem("entity_keys", 0, entity_key)
        add_log(f"Entity key with ID {entity_id} removed from Redis entity_keys list.")

        # Optionally, delete the entity from Supabase
        supabase.table("entities").delete().eq("id", entity_id).execute()
        add_log(f"Entity with ID {entity_id} deleted from Supabase.")

        return {"status": "Entity deleted successfully"}
    except Exception as e:
        # Log and raise any unexpected errors
        error_message = f"Error deleting entity with ID {entity_id}: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)