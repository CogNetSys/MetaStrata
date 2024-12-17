# Archiving Logs to External Hard Drive

To ensure your logs are properly archived and stored, we’ve set up an automated system that moves logs from the server to an external hard drive. Below are the steps to set up this logging archive process.

## Moving Logs to External Drive (Automated)

You can automate the process of moving logs from your server to an external hard drive using a simple bash script. The following script will move the log file to the external drive and append a timestamp to the filename.

### Bash Script Example: `move_log_script.sh`

#!/bin/bash

# Move the log file to the external hard drive
mv /path/to/simulation_app.log /mnt/external_drive/logs/simulation_app_$(date +%Y%m%d).log

This script moves the `simulation_app.log` file to the external drive located at `/mnt/external_drive/logs/`, appending the current date (in `YYYYMMDD` format) to the log filename.

#### Notes:
- Replace `/path/to/simulation_app.log` with the actual path of your log file.
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

### Steps to set up log rotation for your logs:

1. **Create a new log rotation configuration file** for your application by creating a file at `/etc/logrotate.d/simulation_app`.

2. **Logrotate Configuration Example:**

    Create or edit the `/etc/logrotate.d/simulation_app` file with the following content:

    /path/to/simulation_app.log {
        daily                 # Rotate logs daily
        missingok             # Do not error if the log file is missing
        rotate 7              # Keep 7 days worth of logs
        compress              # Compress rotated log files
        notifempty            # Do not rotate if the log is empty
        create 0644 root root # Create a new log file after rotation with proper permissions
    }

    #### Explanation of Logrotate options:
    - `daily`: Rotates the log file every day.
    - `missingok`: Ignores errors if the log file is missing.
    - `rotate 7`: Keeps 7 rotated log files before deleting them.
    - `compress`: Compresses old log files to save space.
    - `notifempty`: Only rotates the log if it’s not empty.
    - `create 0644 root root`: Creates a new log file after rotation with the specified permissions.

# Logging Setup Documentation

## Overview

This project uses a sophisticated logging system that includes **dynamic log level configuration**, **structured JSON logging**, and **asynchronous logging**. The system is designed for easy integration with centralized logging services like **Logstash** and **Elasticsearch**, and it is prepared for future monitoring and alerting integrations.

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
Logs are structured in **JSON** format for easy parsing and integration with systems like **Logstash** and **Elasticsearch**. Each log entry includes:
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
An asynchronous logging handler is used to ensure that logging operations do not block the main application. Log entries are pushed to a queue and processed in a separate thread.

---

## Centralized Logging Setup

The logging system is configured to forward logs to centralized logging systems like **Logstash** or **Elasticsearch**. The logs are structured in JSON format, which makes it easy to process and analyze them using these tools.

### Forwarding Logs to Logstash
To forward your logs to a Logstash server, configure Logstash to receive logs in the structured JSON format. Set up a Logstash input configuration that reads logs from a specific file or stream.

Example Logstash input configuration:
input {
  file {
    path => "/path/to/your/logfile.log"
    start_position => "beginning"
  }
}

---

## Asynchronous Logging and Listener Thread

The logging system includes an asynchronous logging handler that ensures log processing does not block the main application. Logs are placed in a queue and processed by a dedicated listener thread.

### Starting the Listener Thread
To start the asynchronous log listener, a dedicated thread is created that processes log entries in the background.

Example:
import threading

def log_listener():
    while True:
        log_entry = log_queue.get()
        print(log_entry)  # Replace with actual log forwarding logic

# Start listener thread
threading.Thread(target=log_listener, daemon=True).start()

---

## Conclusion

This logging setup ensures that the application has an efficient, scalable, and easy-to-integrate logging system that supports structured logs, asynchronous logging, and dynamic log level management. With the integration of centralized logging tools like Logstash, it is well-positioned for handling large volumes of logs and provides insights into system behavior.
