import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

LOGFIRE_ENABLED = os.getenv("LOGFIRE_ENABLED", "false").lower() == "true"

if LOGFIRE_ENABLED:
    import logfire
    # Configure Logfire before importing other modules
    logfire.configure(environment='local', service_name="CogNetics Architect")
    # logfire.install_auto_tracing(modules=['app'], min_duration=0.01, check_imported_modules='ignore')
    print("Logfire is enabled.")
else:
    print("Logfire is disabled.")

# Import the FastAPI app object after Logfire setup
from app.main import app

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
