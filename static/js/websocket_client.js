const wsUrl = `ws://${window.location.host}/logs`; // WebSocket URL
const websocket = new WebSocket(wsUrl);

function scrollToBottom() {
    const logDiv = document.getElementById("websocket-logs");
    if (logDiv) {
        logDiv.scrollTop = logDiv.scrollHeight;
    }
}

function clearLogs() {
    const logDiv = document.getElementById("websocket-logs");
    if (logDiv) {
        logDiv.innerHTML = "<p>WebSocket logs cleared.</p>";
        scrollToBottom();
    }
}

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

document.addEventListener("DOMContentLoaded", function() {
    const logsDiv = document.getElementById("websocket-logs");

    // Event listener for the clear logs button
    const clearLogsButton = document.getElementById("clear-logs");
    if (clearLogsButton) {
        clearLogsButton.addEventListener("click", function() {
            if (logsDiv) {
                logsDiv.innerHTML = "<p>WebSocket logs will appear here...</p>";
            }
        });
    }

    // Event listener for the select all button
    const selectAllButton = document.getElementById("select-all");
    if (selectAllButton) {
        selectAllButton.addEventListener("click", function() {
            const range = document.createRange();
            range.selectNodeContents(logsDiv);
            const selection = window.getSelection();
            selection.removeAllRanges();
            selection.addRange(range);
        });
    }
});
