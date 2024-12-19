import os
from dotenv import load_dotenv
import logfire
from loguru_integration import setup_loguru
from app.models import PromptSettings, SimulationSettings

# Load environment variables
load_dotenv()

# Set up Loguru logging
setup_loguru()

# Initialize Logfire configuration (before running the app)
LOGFIRE_ENABLED = os.getenv("LOGFIRE_ENABLED", "false").lower() == "true"

if LOGFIRE_ENABLED:
    # Configure Logfire first
    logfire.configure(environment='local', service_name="CogNetics Architect")

    # Install auto-tracing for the 'app' module (or any other modules you want to trace)
    #logfire.install_auto_tracing(modules=['app'], min_duration=0.001, check_imported_modules='ignore')

    print("Logfire is enabled.")
else:
    print("Logfire is disabled.")

# Import the FastAPI app object after Logfire setup
from app.main import app

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
