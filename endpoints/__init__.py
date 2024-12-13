# /endpoints/__init__.py

from fastapi import APIRouter
from .simulation import router as simulation_router
from .worlds import router as worlds_router
from .world_messaging import router as world_messaging_router
from .entities import router as entities_router
from .entity_messaging import router as entity_messaging_router
from .customization import router as customization_router
from .visualizations import router as visualizations_router
from .utilities import router as utilities_router
from .logs import router as logs_router
from .mtnn import router as mtnn_router

# Create a master router
router = APIRouter()

# Include sub-routers
router.include_router(simulation_router, prefix="/simulation", tags=["Simulation"])
router.include_router(worlds_router, prefix="/worlds", tags=["Worlds"])
router.include_router(world_messaging_router, prefix="/world_messaging", tags=["World Messaging"])
router.include_router(entities_router, prefix="/entities", tags=["Entities"]) 
router.include_router(entity_messaging_router, prefix="/entity_messaging", tags=["Entity Messaging"]) 
router.include_router(customization_router, prefix="/customization", tags=["Customization"])
router.include_router(visualizations_router, prefix="/visualization", tags=["Visualization"]) 
router.include_router(utilities_router, prefix="/utilities")
router.include_router(logs_router, prefix="/logs", tags=["Logs"])
router.include_router(mtnn_router, prefix="/mtnn", tags=["mTNN"])
