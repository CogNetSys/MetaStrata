const wsUrl = `ws://localhost:8000/app/ws`;
let websocket = new WebSocket(wsUrl); // WebSocket instance

// Function to scroll to the bottom of the logs div
function scrollToBottom() {
    const logDiv = document.getElementById("websocket-logs");
    if (logDiv) {
        logDiv.scrollTop = logDiv.scrollHeight;
    }
}

// Function to clear logs
function clearLogs() {
    const logDiv = document.getElementById("websocket-logs");
    if (logDiv) {
        logDiv.innerHTML = "<p>WebSocket logs cleared.</p>";
        scrollToBottom();
    }
}

// Function to copy logs to clipboard
function copyLogs() {
    const logDiv = document.getElementById("websocket-logs");
    if (logDiv) {
        const textToCopy = logDiv.innerText || "";
        navigator.clipboard.writeText(textToCopy)
            .then(() => {
                alert("Logs copied to clipboard!");
            })
            .catch(err => {
                console.error("Failed to copy logs:", err);
                alert("Failed to copy logs.");
            });
    }
}

// Function to reconnect the WebSocket stream
function reconnectWebSocket() {
    if (websocket) {
        websocket.close(); // Close the existing connection
    }
    websocket = new WebSocket(wsUrl); // Reconnect WebSocket

    websocket.onopen = () => {
        const logDiv = document.getElementById("websocket-logs");
        if (logDiv) logDiv.innerHTML += "<p>Connected to WebSocket</p>";
        scrollToBottom();
    };

    websocket.onmessage = (event) => {
        const logDiv = document.getElementById("websocket-logs");
        if (logDiv) {
            logDiv.innerHTML += `<p>${event.data}</p>`;
            scrollToBottom();
        }
    };

    websocket.onclose = () => {
        const logDiv = document.getElementById("websocket-logs");
        if (logDiv) logDiv.innerHTML += "<p>Disconnected from WebSocket</p>";
        scrollToBottom();
    };

    websocket.onerror = (error) => {
        const logDiv = document.getElementById("websocket-logs");
        if (logDiv) logDiv.innerHTML += "<p style='color: red;'>WebSocket error</p>";
        scrollToBottom();
    };
}

// WebSocket event listeners
websocket.onopen = () => {
    const logDiv = document.getElementById("websocket-logs");
    if (logDiv) logDiv.innerHTML += "<p>Connected to WebSocket</p>";
    scrollToBottom();
};

websocket.onmessage = (event) => {
    const logDiv = document.getElementById("websocket-logs");
    if (logDiv) {
        logDiv.innerHTML += `<p>${event.data}</p>`;
        scrollToBottom();
    }
};

websocket.onclose = () => {
    const logDiv = document.getElementById("websocket-logs");
    if (logDiv) logDiv.innerHTML += "<p>Disconnected from WebSocket</p>";
    scrollToBottom();
};

websocket.onerror = (error) => {
    const logDiv = document.getElementById("websocket-logs");
    if (logDiv) logDiv.innerHTML += "<p style='color: red;'>WebSocket error</p>";
    scrollToBottom();
};

// Event listener for DOM content loaded
document.addEventListener("DOMContentLoaded", function() {
    const logsDiv = document.getElementById("websocket-logs");

    // Event listener for the clear logs button
    const clearLogsButton = document.getElementById("clear-logs");
    if (clearLogsButton) {
        clearLogsButton.addEventListener("click", clearLogs);
    }

    // Event listener for the copy logs button
    const copyLogsButton = document.getElementById("copy-logs");
    if (copyLogsButton) {
        copyLogsButton.addEventListener("click", copyLogs);
    }

    // Event listener for the reconnect WebSocket button
    const reconnectButton = document.getElementById("reconnect-stream");
    if (reconnectButton) {
        reconnectButton.addEventListener("click", reconnectWebSocket);
    }

    // Adding mouseover effects for buttons
    const buttons = document.querySelectorAll("button");
    buttons.forEach(button => {
        button.addEventListener("mouseover", function() {
            button.style.backgroundColor = "#ddd"; // Change background on hover
            button.style.cursor = "pointer"; // Pointer cursor on hover
        });

        button.addEventListener("mouseout", function() {
            button.style.backgroundColor = ""; // Reset background on mouseout
        });
    });
});