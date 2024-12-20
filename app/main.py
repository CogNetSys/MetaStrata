import httpx
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import SecretStr
from redis.asyncio import Redis
from supabase import create_client, Client
from app.config import settings
from app.database import redis
from app.endpoints import router as endpoints_router
import logfire

# Structured Logging with Logfire
class LogfireHandler(logging.Handler):
    def __init__(self, api_key: SecretStr):
        super().__init__()
        self.api_key = api_key.get_secret_value()
        self.endpoint = "https://logfire.pydantic.dev/cognetsys/cognetics-architect/logs"  # Replace with actual Logfire endpoint

    def emit(self, record):
        log_entry = self.format(record)
        try:
            # Send log to Logfire
            httpx.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                data=log_entry
            )
        except Exception as e:
            print(f"Failed to send log to Logfire: {e}")

# Configure Logger
logger = logging.getLogger('simulation_app')
logger.setLevel(settings.SIMULATION.LOG_LEVEL)
logger.propagate = False  # Prevent duplicate logs

# Initialize logfire_handler as None
logfire_handler = LogfireHandler(settings.LOGFIRE.LOGFIRE_API_KEY)

# Logfire Handler
if settings.LOGFIRE.LOGFIRE_ENABLED and settings.LOGFIRE.LOGFIRE_API_KEY:
    logfire_handler = LogfireHandler(settings.LOGFIRE.LOGFIRE_API_KEY)
    logfire_formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "module": "%(module)s", "line": "%(lineno)d"}'
    )
    logfire_handler.setFormatter(logfire_formatter)
    logger.addHandler(logfire_handler)
    logfire.configure(environment='local', service_name="CogNetics Architect")  # Configure Logfire
else:
    print("Logfire is disabled.")

# **Additional Handlers (Optional)**
# You can add other handlers like console or file handlers if needed
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter(
    '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "module": "%(module)s", "line": "%(lineno)d"}'
)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Initialize the custom logger for patched logs
logfire_debug_logger = logging.getLogger("logfire_debug")
logfire_debug_logger.setLevel(logging.DEBUG)
logfire_debug_logger.propagate = False  # Prevent propagation to root logger
logfire_debug_handler = logging.StreamHandler()  # Log to the console
logfire_debug_formatter = logging.Formatter(
    '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logfire_debug_handler.setFormatter(logfire_debug_formatter)
logfire_debug_logger.addHandler(logfire_debug_handler)

# Stop signal
stop_signal = False

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

# **Inspector Integration (Assuming a Monitoring Tool)**
# Example: Using FastAPI's built-in middleware or a third-party tool like Prometheus
# Here, we'll add a simple middleware for request inspection

app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["*"]
)
