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

    # Add Loguru handler to log to a file
    logger.add("logs/application.log", rotation="10 MB", retention="10 days", compression="zip", backtrace=True, diagnose=True)

    # Optionally, log to console
    logger.add(sys.stdout, format="{time} {level} {message}", level="DEBUG")

    # Attach InterceptHandler to Python's root logger
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.DEBUG)

    # Optional: Redirect specific loggers to Loguru
    logging.getLogger("fastapi").handlers = [InterceptHandler()]
    logging.getLogger("uvicorn").handlers = [InterceptHandler()]
