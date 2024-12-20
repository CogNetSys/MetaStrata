# /app/endpoints/entities.py

import logfire
from app.config import Settings, settings, calculate_chebyshev_distance
from app.utils.database import redis, supabase
from app.utils.models import Entity
from fastapi import APIRouter, HTTPException
from typing import List

router = APIRouter()

@router.post("/entities", response_model=Entity, tags=["Entities"])
async def create_entity(entity: Entity):
    await redis.hset(f"entity:{entity.id}", mapping=entity.dict())
    if settings.LOGFIRE.LOGFIRE_ENABLED:
        logfire.info(f"Entity {entity.id} created in Redis.")
    
    supabase.table("entities").insert(entity.dict()).execute()
    if settings.LOGFIRE.LOGFIRE_ENABLED:
        logfire.info(f"Entity {entity.id} stored in Supabase.")
        
    return entity

@router.get("/entities/{entity_id}", response_model=Entity, tags=["Entities"])
async def get_entity(entity_id: int):
    entity_data = await redis.hgetall(f"entity:{entity_id}")
    if not entity_data:
        if settings.LOGFIRE.LOGFIRE_ENABLED:
            logfire.error(f"Entity {entity_id} not found.")
        raise HTTPException(status_code=404, detail="Entity not found")

    try:
        entity = Entity(
            id=int(entity_data["id"]),
            name=entity_data["name"],
            x=int(entity_data["x"]),
            y=int(entity_data["y"]),
            memory=entity_data.get("memory", "")
        )
        if settings.LOGFIRE.LOGFIRE_ENABLED:
            logfire.debug(f"Entity {entity_id} retrieved from Redis.")
        return entity
    except (KeyError, ValueError) as e:
        if settings.LOGFIRE.LOGFIRE_ENABLED:
            logfire.error(f"Error parsing entity {entity_id} data: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving entity data")

@router.put("/entities/{entity_id}", response_model=Entity, tags=["Entities"])
async def update_entity(entity_id: int, entity: Entity):
    await redis.hset(f"entity:{entity_id}", mapping=entity.dict())
    if settings.LOGFIRE.LOGFIRE_ENABLED:
        logfire.info(f"Entity {entity_id} updated in Redis.")

    supabase.table("entities").update(entity.dict()).eq("id", entity_id).execute()
    if settings.LOGFIRE.LOGFIRE_ENABLED:
        logfire.info(f"Entity {entity_id} updated in Supabase.")
        
    return entity

@router.post("/entities/{entity_id}/send_message", tags=["Entities"])
async def send_message(entity_id: int, message: str):
    entity_data = await redis.hgetall(f"entity:{entity_id}")
    if not entity_data:
        if settings.LOGFIRE.LOGFIRE_ENABLED:
            logfire.error(f"Entity {entity_id} not found for sending message.")
        raise HTTPException(status_code=404, detail="Entity not found")
    
    await redis.hset(f"entity:{entity_id}", "message", message)
    if settings.LOGFIRE.LOGFIRE_ENABLED:
        logfire.info(f"Message sent by Entity {entity_id}: {message}")
    
    return {"status": "Message sent successfully", "message": message}

@router.delete("/entities/{entity_id}", tags=["Entities"])
async def delete_entity(entity_id: int):
    deleted = await redis.delete(f"entity:{entity_id}")
    if deleted:
        if settings.LOGFIRE.LOGFIRE_ENABLED:
            logfire.info(f"Entity {entity_id} deleted from Redis.")
    else:
        if settings.LOGFIRE.LOGFIRE_ENABLED:
            logfire.error(f"Attempted to delete non-existent Entity {entity_id} from Redis.")
    
    supabase.table("entities").delete().eq("id", entity_id).execute()
    if settings.LOGFIRE.LOGFIRE_ENABLED:
        logfire.info(f"Entity {entity_id} deleted from Supabase.")
    
    return {"status": "Entity deleted successfully"}

@router.get("/entities/{entity_id}/nearby", response_model=List[Entity], tags=["Entities"])
async def get_nearby_entities(entity_id: int):
    entity_data = await redis.hgetall(f"entity:{entity_id}")
    if not entity_data:
        if settings.LOGFIRE.LOGFIRE_ENABLED:
            logfire.error(f"Entity {entity_id} not found for fetching nearby entities.")
        raise HTTPException(status_code=404, detail="Entity not found")

    try:
        entity = Entity(
            id=int(entity_data["id"]),
            name=entity_data["name"],
            x=int(entity_data["x"]),
            y=int(entity_data["y"]),
            memory=entity_data.get("memory", "")
        )
    except (KeyError, ValueError) as e:
        if settings.LOGFIRE.LOGFIRE_ENABLED:
            logfire.error(f"Error parsing entity {entity_id} data: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving entity data")

    all_entities = []
    for i in range(settings.SIMULATION.NUM_ENTITIES):
        if i != entity_id:
            entity_info = await redis.hgetall(f"entity:{i}")
            if entity_info and all(k in entity_info for k in ["id", "name", "x", "y"]):
                try:
                    nearby_entity = Entity(
                        id=int(entity_info["id"]),
                        name=entity_info["name"],
                        x=int(entity_info["x"]),
                        y=int(entity_info["y"]),
                        memory=entity_info.get("memory", "")
                    )
                    all_entities.append(nearby_entity)
                except (KeyError, ValueError) as e:
                    if settings.LOGFIRE.LOGFIRE_ENABLED:
                        logfire.error(f"Missing or invalid data for entity {i}: {e}. Skipping.")
            else:
                if settings.LOGFIRE.LOGFIRE_ENABLED:
                    logfire.error(f"Missing or incomplete data for entity {i}. Skipping.")

    nearby_entities = [
        a for a in all_entities
        if calculate_chebyshev_distance(entity.x, entity.y, a.x, a.y) <= settings.SIMULATION.CHEBYSHEV_DISTANCE
    ]

    if settings.LOGFIRE.LOGFIRE_ENABLED:
        logfire.debug(f"Fetched {len(nearby_entities)} nearby entities for Entity {entity_id}.")
    return nearby_entities

@router.post("/sync_entities", tags=["Entities"])
async def sync_entities():
    all_entities = []
    for i in range(settings.SIMULATION.NUM_ENTITIES):
        entity_data = await redis.hgetall(f"entity:{i}")
        if entity_data and all(k in entity_data for k in ["id", "name", "x", "y"]):
            try:
                entity = Entity(
                    id=int(entity_data["id"]),
                    name=entity_data["name"],
                    x=int(entity_data["x"]),
                    y=int(entity_data["y"]),
                    memory=entity_data.get("memory", "")
                )
                all_entities.append(entity)
            except (KeyError, ValueError) as e:
                if settings.LOGFIRE.LOGFIRE_ENABLED:
                    logfire.error(f"Error parsing entity {i} data: {e}. Skipping.")
        else:
            if settings.LOGFIRE.LOGFIRE_ENABLED:
                logfire.error(f"Missing or incomplete data for entity {i}. Skipping.")

    for entity in all_entities:
        supabase.table("entities").upsert(entity.dict()).execute()
        if settings.LOGFIRE.LOGFIRE_ENABLED:
            logfire.info(f"Entity {entity.id} synchronized to Supabase.")

    if settings.LOGFIRE.LOGFIRE_ENABLED:
        logfire.info("All entities synchronized between Redis and Supabase.")
    return {"status": "Entities synchronized between Redis and Supabase"}
