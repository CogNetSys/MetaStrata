# /run.py

import logfire
from logfire_integration import setup_loguru
from app.config import Settings, settings

# # Print the entire settings object for debugging
# print("--- Full settings object: ---")

# # Iterate over the attributes of the settings object
# for key, value in settings.__dict__.items():
#     if hasattr(value, 'model_dump_json'):
#         print(f"--- {key}: ---")
#         print(value.model_dump_json(indent=2))
#     else:
#         print(f"{key}: {value}")

# print("--- End of settings object ---")

# # Print the specific LOGFIRE_ENDPOINT
# print(f"LOGFIRE_ENDPOINT from settings: {settings.LOGFIRE.LOGFIRE_ENDPOINT}")

# Set up Loguru logging
setup_loguru()

# Initialize Logfire configuration (before running the app)
if settings.LOGFIRE.logfire_enabled:
    logfire.configure(
        token=settings.LOGFIRE.logfire_api_key,  # Use token for API key
        environment="local",
        service_name="CogNetics Architect",
    )

    # Keep the line below, use it for performance metrics.
    # logfire.instrument_system_metrics(base='full')

    # Install auto-tracing for the 'app' module (or any other modules you want to trace)
    # logfire.install_auto_tracing(modules=['app'], min_duration=0.01, check_imported_modules='ignore')

    print(f"Printing Logfire URL: {settings.LOGFIRE.LOGFIRE_ENDPOINT}")
    print("Logfire is enabled.")
else:
    print("Logfire is disabled.")

# Import the FastAPI app object after Logfire setup
from app.main import app

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
