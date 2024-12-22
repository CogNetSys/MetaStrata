# /app/main.py

import logfire
from app.config import settings
from app.utils.logging import setup_logging
from app.utils.database import redis
from app.endpoints import router as endpoints_router
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Stop signal
stop_signal = False

# LOGGING SETUP
logger = setup_logging()

# ----------------------------------------------
# LIFESPAN SECTION
# ----------------------------------------------

# Lifespan Context Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    global stop_signal
    if settings.LOGFIRE.LOGFIRE_ENABLED:
        logfire.info('Starting application lifespan...')
    
    try:
        # Startup logic
        if settings.LOGFIRE.LOGFIRE_ENABLED:
            logfire.info('Application started!')
        await redis.ping()
        if settings.LOGFIRE.LOGFIRE_ENABLED:
            logfire.info('Redis connection established.')       
        yield
    finally:
        # Shutdown logic
        if settings.LOGFIRE.LOGFIRE_ENABLED:
            logfire.info('Shutting down application...')
        stop_signal = True  # Ensure simulation stops if running
        await redis.close()
        if settings.LOGFIRE.LOGFIRE_ENABLED:
            logfire.info('Redis connection closed.')

# ----------------------------------------------
# FASTAPI SECTION
# ----------------------------------------------

# Create FastAPI app with lifespan
app = FastAPI(
    title="CogNetics Architect", 
    version="0.0.3", 
    description="This is the enhanced implementation of the <a href='https://arxiv.org/pdf/2411.03252' target='_blank'>Takata et al experiment.</a>",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(endpoints_router)