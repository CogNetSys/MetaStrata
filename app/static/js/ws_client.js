document.addEventListener('DOMContentLoaded', () => {
    const logContainer = document.getElementById('logContainer');
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/logs`;
    
    const socket = new WebSocket(wsUrl);

    socket.onopen = () => {
        console.log('WebSocket connection established');
        logContainer.innerHTML += '<p>WebSocket connected</p>';
    };

    socket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'heartbeat') {
                console.log('Heartbeat received');
                return;
            }
            logContainer.innerHTML += `<p>${event.data}</p>`;
            logContainer.scrollTop = logContainer.scrollHeight;
        } catch (error) {
            console.error('Error parsing message:', error);
            logContainer.innerHTML += `<p>Raw message: ${event.data}</p>`;
        }
    };

    socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        logContainer.innerHTML += `<p>WebSocket error: ${error}</p>`;
    };

    socket.onclose = () => {
        console.log('WebSocket connection closed');
        logContainer.innerHTML += '<p>WebSocket disconnected</p>';
    };
});