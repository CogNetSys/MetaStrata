import logging
import httpx
from pydantic import SecretStr
from app.config import settings
import logfire

# Structured Logging with Logfire
class LogfireHandler(logging.Handler):
    def __init__(self, api_key: SecretStr):
        super().__init__()
        self.api_key = api_key.get_secret_value()
        self.endpoint = "https://logfire.pydantic.dev/cognetsys/cognetics-architect/logs"

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

def setup_logging():
    logger = logging.getLogger('simulation_app')
    logger.setLevel(settings.SIMULATION.LOG_LEVEL)
    logger.propagate = False  # Prevent duplicate logs

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

    # Console Handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "module": "%(module)s", "line": "%(lineno)d"}'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger
