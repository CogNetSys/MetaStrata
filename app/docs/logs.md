# Logging Setup Documentation

## Overview

This project uses a sophisticated logging system that includes **dynamic log level configuration**, **structured JSON logging**, and **asynchronous logging**. The system is designed for easy integration with centralized logging services like **Logfire** and **Loguru**, and it is prepared for future monitoring and alerting integrations.

### Key Features:
- **Dynamic Log Level**: Change log levels via the `LOG_LEVEL` environment variable (e.g., `DEBUG`, `INFO`, `ERROR`).
- **Structured Logging**: Logs are captured in JSON format, making it easier for downstream systems to parse and analyze logs.
- **Asynchronous Logging**: Log entries are processed asynchronously, ensuring that the main application flow is not blocked by logging operations.

---

## Logging Configuration

### Environment Configuration
To control the logging level, you can set the `LOG_LEVEL` environment variable. Valid values are:
- `DEBUG`
- `INFO`
- `WARNING`
- `ERROR`
- `CRITICAL`

If the `LOG_LEVEL` environment variable is not set, it defaults to `INFO`.

Example:
export LOG_LEVEL=DEBUG

### JSON Structured Logging
Logs are structured in **JSON** format for easy parsing and integration with systems like **Logfire**. Each log entry includes:
- `timestamp`: The time the log was generated.
- `level`: The log level (e.g., `INFO`, `DEBUG`).
- `message`: The actual log message.
- `module`: The module where the log was generated.
- `line`: The line number of the log entry.

Example log entry:
{
    "timestamp": "2024-12-16T14:00:00Z",
    "level": "INFO",
    "message": "Entities initialized.",
    "module": "simulation_app",
    "line": 140
}

### Asynchronous Logging
An asynchronous logging handler is used to ensure that logging operations do not block the main application. Log entries are placed in a queue and processed by a dedicated listener thread.

---

## Centralized Logging Setup

The logging system is configured to forward logs to **Logfire**. The logs are structured in JSON format, which makes it easy to process and analyze them.

### Forwarding Logs to Logfire
To forward your logs to Logfire, configure Logfire to receive logs in the structured JSON format.

---

## Asynchronous Logging and Listener Thread

The logging system includes an asynchronous logging handler that ensures log processing does not block the application.

### Logfire Integration

The `LogfireHandler` class integrates with the Python `logging` module to forward log entries to Logfire. The `LogfireHandler` is responsible for formatting and sending logs to the Logfire endpoint.

Example Logfire handler:
class LogfireHandler(logging.Handler):
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
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
            print(f"Error sending log to Logfire: {e}")

---

## Moving Logs to External Drive (Automated)

You can automate the process of moving logs from your server to an external hard drive using a simple bash script. The following script will move the log file to the external drive and append a timestamp to the filename.

### Bash Script Example: `move_log_script.sh`

#!/bin/bash

# Move the log file to the external hard drive
mv /path/to/application.log /mnt/external_drive/logs/application_$(date +%Y%m%d).log

This script moves the `application.log` file to the external drive located at `/mnt/external_drive/logs/`, appending the current date (in `YYYYMMDD` format) to the log filename.

#### Notes:
- Replace `/path/to/application.log` with the actual path of your log file.
- Replace `/mnt/external_drive/logs/` with the actual path of your mounted external hard drive directory.

## Setting up a Cron Job

You can set up a **cron job** to run this script periodically (e.g., daily at midnight) to automate the archiving process. 

### Steps to set up a cron job:

1. Open your crontab file for editing by running the following command:
    crontab -e

2. Add the following line to the crontab file to execute the `move_log_script.sh` every day at midnight:

    0 0 * * * /path/to/your/move_log_script.sh

    This line tells cron to run the script every day at 12:00 AM. You can adjust the timing as needed.

## Log Rotation (Optional)

To manage log files and prevent them from growing indefinitely, you can set up **log rotation**. Log rotation allows you to compress, archive, or delete old log files after a certain period.
