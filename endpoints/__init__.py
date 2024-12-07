# /endpoints/__init__.py

from fastapi import APIRouter
from .worlds import router as worlds_router
from .customization import router as customization_router
from .visualizations import router as visualizations_router
from .entities import router as entities_router
from .utilities import router as utilities_router
from .logs import router as logs_router

# Create a master router
router = APIRouter()

# Include sub-routers
router.include_router(worlds_router, prefix="/worlds", tags=["World Simulation"])
router.include_router(customization_router, prefix="/customization", tags=["Customization"])
router.include_router(visualizations_router, prefix="/visualization", tags=["Visualization"]) 
router.include_router(entities_router, prefix="/entities", tags=["Entities"]) 
router.include_router(utilities_router, prefix="/utilities")
router.include_router(logs_router, prefix="/logs", tags=["Logs"])