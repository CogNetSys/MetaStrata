from fastapi import APIRouter
from .settings import router as settings_router
from .prompts import router as prompts_router
from .worlds import router as worlds_router

# Create a master router
router = APIRouter()

# Include sub-routers
router.include_router(settings_router, prefix="/settings", tags=["Settings"])
router.include_router(prompts_router, prefix="/prompts", tags=["Prompts"])
router.include_router(worlds_router, prefix="/worlds", tags=["World Simulation"])
