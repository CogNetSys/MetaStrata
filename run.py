import logfire

# Configure Logfire
logfire.configure()
logfire.install_auto_tracing(modules=['app'], min_duration=0.01)

# Import the FastAPI app object
from app.main import app

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
