from fastapi import APIRouter
from .simulation import router as simulation_router
from .customization import router as customization_router
from .entities import router as entities_router

# Initialize a consolidated APIRouter
router = APIRouter()

# Include other routers without specific prefixes
router.include_router(simulation_router)
router.include_router(customization_router)
router.include_router(entities_router)
