# /logfire_integration.py

from loguru import logger
import logging
import sys

class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Map Python logging levels to Loguru levels
        level = record.levelname if record.levelname in logger._core.levels else record.levelno

        # Log the message to Loguru
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())

def setup_loguru():
    # Remove default Loguru handlers to avoid double logging
    logger.remove()

    # Loguru handler for the application log (using text/record format)
    logger.add(
        "logs/application.log",
        serialize=True,
        rotation="10 MB",
        retention="10 days",
        compression="zip",
        backtrace=True,
        diagnose=True
    )
 
    # Log to console
    logger.add(sys.stdout, format="{time} {level} {message}", level="DEBUG")

    # Attach InterceptHandler to the root logger ONLY
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.DEBUG)

    # Prevent uvicorn and fastapi logs from being handled by the root logger (and thus InterceptHandler)
    logging.getLogger("fastapi").propagate = False
    logging.getLogger("uvicorn").propagate = False
