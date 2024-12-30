import logging
import httpx
import logfire
from pydantic import SecretStr

# Structured Logging with Logfire
class LogfireHandler(logging.Handler):
    def __init__(self, api_key: SecretStr, endpoint: str):
        super().__init__()
        self.api_key = api_key.get_secret_value()
        self.endpoint = endpoint

    def emit(self, record):
        log_entry = self.format(record)
        print(f"Logfire Handler Endpoint: {self.endpoint}")
        try:
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

def setup_logging():
    # Lazy import of settings to avoid circular dependency
    from app.config import settings

    logger = logging.getLogger('simulation_app')
    logger.setLevel(settings.SIMULATION.LOG_LEVEL)
    logger.propagate = False

    # Logfire Handler
    if settings.LOGFIRE.LOGFIRE_ENABLED and settings.LOGFIRE.LOGFIRE_API_KEY:
        logfire.configure(
            token=settings.LOGFIRE.logfire_api_key,  # Use token for API key
            environment='local',
            service_name="CogNetics Architect",

        )
        logfire_handler = LogfireHandler(settings.LOGFIRE.LOGFIRE_API_KEY, settings.LOGFIRE.LOGFIRE_ENDPOINT)
        logfire_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "module": "%(module)s", "line": "%(lineno)d"}'
        )
        logfire_handler.setFormatter(logfire_formatter)
        logger.addHandler(logfire_handler)
    else:
        print("Logfire is disabled.")

    # Console Handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "module": "%(module)s", "line": "%(lineno)d"}'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger
