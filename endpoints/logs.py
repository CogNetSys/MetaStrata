import logging
import os
from fastapi import APIRouter, HTTPException
from logging.handlers import RotatingFileHandler
from config import LOG_FILE, LOG_DIR, LOG_QUEUE_MAX_SIZE
from utils import add_log, LOG_QUEUE

router = APIRouter()

# Global logger instance
logger = logging.getLogger("simulation_app")

# Set up rotating file handler and log level dynamically
def setup_rotating_handler():
    """Setup rotating handler with the current maxBytes and backupCount."""
    return RotatingFileHandler(
        LOG_FILE, mode='a', maxBytes=10**6, backupCount=3
    )

@router.get("/audit")
async def fetch_audit_trail():
    """
    Fetch a detailed log of all significant actions performed during the simulation.

    This function aggregates logs from the in-memory queue (`LOG_QUEUE`) 
    and all log files within the configured `LOG_DIR`.
    """
    try:
        audit_logs = []

        # Add logs from the in-memory queue
        if LOG_QUEUE:
            audit_logs.extend(list(LOG_QUEUE))

        # Add logs from all files in the log directory
        if os.path.exists(LOG_DIR):
            for file_name in sorted(os.listdir(LOG_DIR)):  # Sort files by name for consistent order
                file_path = os.path.join(LOG_DIR, file_name)
                if os.path.isfile(file_path):
                    with open(file_path, "r") as file:
                        audit_logs.extend(file.readlines())
        else:
            add_log("Log directory does not exist.")

        if not audit_logs:
            add_log("Audit trail requested, but no logs found.")
            return {"audit_trail": []}

        return {"audit_trail": audit_logs}
    except Exception as e:
        error_message = f"Error fetching audit trail: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)
    
@router.post("/logs_reset")
async def reset_logs():
    """
    <b>Reset both the in-memory log queue and the log file.</b>

    This endpoint resets the logging system, clearing any accumulated logs and resetting the log queue. It is useful for cases where logs need to be cleared out before starting a fresh set of logs or to free up resources.

    Usage:

        Method: POST
        Request: Trigger a reset of the logs (no additional body required).
        Response: Success message if logs were successfully reset.
    """
    try:
        # Clear the in-memory queue (LOG_QUEUE)
        LOG_QUEUE.clear()

        # Clear all log files in the LOG_DIR
        if os.path.exists(LOG_DIR):
            for file_name in os.listdir(LOG_DIR):
                file_path = os.path.join(LOG_DIR, file_name)
                if os.path.isfile(file_path):
                    with open(file_path, 'w') as file:
                        file.truncate(0)  # Clear file contents
        else:
            add_log("Log directory does not exist during reset.")
            return {"status": "Log directory does not exist."}

        # Add log message for the reset action
        add_log("Logs have been reset (queue and files).")
        return {"status": "Logs reset successfully."}
    except Exception as e:
        error_message = f"Error resetting logs: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@router.get("/logs/log_queue_size")
async def log_queue_size():
    """
    <b>Fetch the current maximum size (maxlen) of the log queue.</b>

    This endpoint retrieves or adjusts the current size of the log queue, which defines how many log entries are retained in memory before older entries are discarded. This is useful for managing memory usage in long-running applications where logs can accumulate over time.
    
    Big Picture:

    Managing the log queue size ensures that memory usage remains within reasonable limits. By adjusting the queue size, users can balance between keeping detailed logs and conserving memory. This is especially useful for real-time monitoring or systems with high log volumes.
    
    Usage:

        Method: GET to retrieve the current size or POST to adjust the size.
        Request: To adjust, send the desired maxlen value.
        Response: The current or updated log queue size.
    """
    try:
        # Return the current maxlen of the log queue
        return {"current_log_queue_size": LOG_QUEUE.maxlen}
    except Exception as e:
        error_message = f"Error fetching log queue size: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

# Endpoint to change the maxlen of the log queue
@router.post("/logs/size", tags=["Utilities"])
async def set_log_queue_size(new_maxlen: int):
    """
    Adjust the maximum size of the log queue.

    Description:

    This endpoint retrieves the size of the current log file(s). It can help users monitor the log storage usage and decide when logs need to be archived or rotated. This is especially important in systems that generate large volumes of logs, to prevent disk space from being consumed excessively.

    Big Picture:

    Log file size monitoring is crucial in systems where logs are written continuously. Knowing the log size helps with managing disk space and implementing log rotation policies effectively.

    Usage:

        Method: GET
        Request: No body required.
        Response: The current size of the log file(s) in bytes or human-readable format (e.g., MB or GB).
    """
    try:
        if new_maxlen < 1:
            raise HTTPException(status_code=400, detail="maxlen must be a positive integer.")
        
        # Adjust the log queue size dynamically
        global LOG_QUEUE  # Declare the global LOG_QUEUE to modify it
        
        # Preserve current logs (copy them) and reset the queue
        current_logs = list(LOG_QUEUE)  # Copy current logs
        LOG_QUEUE.clear()  # Clear the existing queue
        
        # Reinitialize the log queue with the new maxlen
        LOG_QUEUE = deque(current_logs, maxlen=new_maxlen)

        add_log(f"Log queue size adjusted to {new_maxlen}.")
        return {"status": "Log queue size updated successfully", "new_maxlen": new_maxlen}
    except Exception as e:
        error_message = f"Error updating log queue size: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@router.get("/logs/level")
async def get_log_level():
    """
    <b>Retrieve the current log level of the logging system.</b>
    
    The log level controls the verbosity of the logs generated by the application. 
    Possible values are: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
    
    This is useful for monitoring or ensuring that the correct level is being used 
    for the current environment (e.g., more detailed logs for debugging, fewer logs 
    for production).

    <ul>
        <li><strong>DEBUG</strong>: 
            <p>This is the most granular logging level. It's used for detailed diagnostic output, useful during development. Debug logs typically contain information about the flow of the program, variable values, and internal states. These logs should provide the most insight into the application's internal workings.</p>
        </li>
        <li><strong>INFO</strong>: 
            <p>Informational messages are used to report normal application operation. These logs can indicate the completion of a task or significant milestones, such as the start of a server, the processing of data, or other important events. The logs are generally used for tracking the general progress of the application.</p>
        </li>
        <li><strong>WARNING</strong>: 
            <p>Warnings are used to signal that something unexpected occurred, but it does not prevent the application from continuing to function. These logs indicate a potential issue that could cause problems later but is not critical at the moment. For example, when an expected file is missing, but the application can still proceed with default behavior.</p>
        </li>
        <li><strong>ERROR</strong>: 
            <p>Error logs represent a more serious problem, usually indicating that a part of the program failed to execute as expected. The error is likely to prevent some functionality from working correctly, but the application might still continue running. These logs typically represent failures or issues that need attention but aren't fatal.</p>
        </li>
        <li><strong>CRITICAL</strong>: 
            <p>Critical logs represent the highest severity. They indicate a catastrophic failure that requires immediate attention. This could include system crashes, database failures, or other situations where the application cannot continue running properly. Critical logs are essential for urgent debugging and typically signal that intervention is needed as soon as possible.</p>
        </li>
    </ul>
    """
    try:
        log_level = logging.getLevelName(logger.level)  # Convert level number to human-readable name
        return {"current_log_level": log_level}
    except Exception as e:
        error_message = f"Error fetching log level: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@router.post("/logs/level")
async def set_log_level(level: str):
    """
    <b>Set the logging level dynamically.</b>

    This endpoint allows the log level to be adjusted dynamically. Log levels are DEBUG, INFO, WARNING, ERROR, and CRITICAL. This is useful for controlling the verbosity of logs in real-time, particularly when debugging issues or in production environments where you may want to reduce log noise.

    By allowing dynamic log level changes, this endpoint enables better control over logging output, which can be crucial in debugging or monitoring live systems. For example, setting the log level to DEBUG in development can provide detailed logs, while setting it to WARNING or ERROR in production can reduce the volume of logs generated.

    <b>Usage:</b>

        Method: GET to retrieve the current log level, POST to change the level.
        Request: To change the level, send a JSON body with the new log level.
        Response: The current or updated log level.

    <ul>
        <li><strong>DEBUG</strong>: 
            <p>This is the most granular logging level. It's used for detailed diagnostic output, useful during development. Debug logs typically contain information about the flow of the program, variable values, and internal states. These logs should provide the most insight into the application's internal workings.</p>
        </li>
        <li><strong>INFO</strong>: 
            <p>Informational messages are used to report normal application operation. These logs can indicate the completion of a task or significant milestones, such as the start of a server, the processing of data, or other important events. The logs are generally used for tracking the general progress of the application.</p>
        </li>
        <li><strong>WARNING</strong>: 
            <p>Warnings are used to signal that something unexpected occurred, but it does not prevent the application from continuing to function. These logs indicate a potential issue that could cause problems later but is not critical at the moment. For example, when an expected file is missing, but the application can still proceed with default behavior.</p>
        </li>
        <li><strong>ERROR</strong>: 
            <p>Error logs represent a more serious problem, usually indicating that a part of the program failed to execute as expected. The error is likely to prevent some functionality from working correctly, but the application might still continue running. These logs typically represent failures or issues that need attention but aren't fatal.</p>
        </li>
        <li><strong>CRITICAL</strong>: 
            <p>Critical logs represent the highest severity. They indicate a catastrophic failure that requires immediate attention. This could include system crashes, database failures, or other situations where the application cannot continue running properly. Critical logs are essential for urgent debugging and typically signal that intervention is needed as soon as possible.</p>
        </li>
    </ul>
    """
    try:
        level = level.upper()
        if level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            raise HTTPException(status_code=400, detail="Invalid log level. Choose from DEBUG, INFO, WARNING, ERROR, or CRITICAL.")
        
        # Update the logger level
        logger.setLevel(getattr(logging, level))

        # Update the log level for all handlers
        for handler in logger.handlers:
            handler.setLevel(getattr(logging, level))
        
        add_log(f"Logging level dynamically updated to {level}.")
        return {"status": f"Logging level set to {level}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting log level: {str(e)}")

@router.post("/logs/settings")
async def set_log_file_settings(new_max_bytes: int, new_backup_count: int):
    """
    <b>Set the maxBytes and backupCount for the log file dynamically.</b>

    Description:

    This endpoint provides or modifies the logging settings, including things like log level, format, and whether logging is enabled. It's useful for configuring the logging system without needing to modify the application code directly.

    Big Picture:

    This endpoint is a key part of log management in a dynamic system, allowing administrators or users to adjust how logs are handled at runtime. This can be important for troubleshooting or adapting to different environments (e.g., development vs. production).

    Usage:

        Method: GET to retrieve the current settings or POST to update the settings.
        Request: To update settings, send a JSON body with desired changes.
        Response: The current or updated logging settings.
    """
    global logger  # Access the logger to adjust its handler
    try:
        if new_max_bytes <= 0 or new_backup_count < 0:
            raise HTTPException(status_code=400, detail="Invalid values for max_bytes or backup_count.")
        
        # Update the global variables for max_bytes and backup_count
        max_bytes = new_max_bytes
        backup_count = new_backup_count

        # Reinitialize the rotating file handler with new settings
        rotating_handler = RotatingFileHandler(LOG_FILE, mode='a', maxBytes=new_max_bytes, backupCount=new_backup_count)
        rotating_handler.setLevel(logging.INFO)

        # Replace the current handler with the new one
        for handler in logger.handlers:
            if isinstance(handler, RotatingFileHandler):
                logger.removeHandler(handler)  # Remove old handler
        logger.addHandler(rotating_handler)  # Add new handler

        add_log(f"Log file settings updated: max_bytes={new_max_bytes}, backup_count={new_backup_count}.")
        return {"status": "Log settings updated successfully", "max_bytes": new_max_bytes, "backup_count": new_backup_count}
    except Exception as e:
        error_message = f"Error updating log file settings: {str(e)}"
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@router.get("/logs/size")
async def get_log_file_size():
    """
    Get the current maxBytes setting for the log file.
    """
    return {"max_bytes": max_bytes, "backup_count": backup_count}

@router.post("/logs/size")
async def set_log_file_size(new_max_bytes: int, new_backup_count: int):
    """
    Set the maxBytes and backupCount for the log file.
    """
    global max_bytes, backup_count  # Declare global here before modifying them

    try:
        if new_max_bytes <= 0 or new_backup_count < 0:
            raise HTTPException(status_code=400, detail="Invalid values for max_bytes or backup_count.")
        
        # Update the global variables
        max_bytes = new_max_bytes
        backup_count = new_backup_count
        
        # Reinitialize the rotating handler with new settings
        handler = setup_rotating_handler()
        logger.handlers = [handler]  # Replace the existing handler with the new one
        
        return {"status": "Log settings updated successfully.", "max_bytes": max_bytes, "backup_count": backup_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
