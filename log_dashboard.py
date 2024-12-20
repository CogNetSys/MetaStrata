# /log_dashboard.py

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import plotly.express as px
from collections import deque
import random
import datetime
import re
import os
import numpy as np
from wordcloud import WordCloud

# Updated parse_log_line function to handle extra space after timestamp
def parse_log_line(line):
    """Parses a log line into a dictionary."""
    parts = line.split("|")  # Split by "|" instead of " | "
    if len(parts) < 4:
        return None

    timestamp_str = parts[0].strip()  # Remove leading/trailing spaces
    level = parts[1].strip()
    message_type = parts[2].strip()
    message = parts[3].strip()

    try:
        timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return None

    log_data = {
        "timestamp": timestamp,
        "level": level,
        "message_type": message_type,
        "message": message,
    }
    return log_data

# Function to generate word cloud (no changes needed)
def generate_wordcloud(word_counts):
    """Generates a word cloud from word counts."""
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)
    return wordcloud.to_array()

# Initialize data structures for visualization
max_data_points = 100
entity_positions = {i: deque(maxlen=max_data_points) for i in range(3)}
message_counts = deque(maxlen=max_data_points)
timestamps = deque(maxlen=max_data_points)
words_in_messages = []
http_requests = deque(maxlen=max_data_points)
grid_size = 15

# Initialize the Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div(
    [
        html.H1("LLM Agent Simulation Dashboard"),
        dcc.Graph(id="entity-positions-heatmap"),
        dcc.Graph(id="entity-positions-graph"),
        dcc.Graph(id="message-word-cloud"),
        dcc.Graph(id="message-counts-graph"),
        dcc.Graph(id="http-requests-graph"),
        dcc.Interval(
            id="interval-component", interval=30 * 1000, n_intervals=0
        ),
    ]
)

# Callback to update visualizations (with updated parsing logic)
@app.callback(
    [
        Output("entity-positions-heatmap", "figure"),
        Output("entity-positions-graph", "figure"),
        Output("message-word-cloud", "figure"),
        Output("message-counts-graph", "figure"),
        Output("http-requests-graph", "figure")
    ],
    [Input("interval-component", "n_intervals")],
)
def update_dashboard(n):
    """Updates the dashboard with new data from the log file."""
    global entity_positions, message_counts, timestamps, words_in_messages, http_requests

    # Define log_file_path at the beginning of the function
    log_file_path = "/home/irbsurfer/Projects/MetaStrata/logs/application.log"

    # Read and process new log data
    try:
        with open(log_file_path, "r") as f:
            log_content = f.read()
        print(f"Successfully read log file: {log_file_path}")
        print(f"--- Log Content Start ---")
        print(log_content)
        print(f"--- Log Content End ---")
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        log_content = ""
    except Exception as e:
        print(f"Error reading log file: {e}")
        log_content = ""

    log_lines = log_content.split("\n")
    print(f"Number of log lines: {len(log_lines)}")

    new_messages = 0

    for line_number, line in enumerate(log_lines):
        print(f"Processing line {line_number + 1}: {line}")
        log_data = parse_log_line(line)
        if not log_data:
            print(f"  Could not parse line {line_number + 1}")
            continue

        # Parsing logic (no changes here)
        # ...

    timestamps.append(datetime.datetime.now())
    message_counts.append(new_messages)

    # Create figures
    # 1. Entity Positions Heatmap
    heatmap_data = np.zeros((grid_size, grid_size))
    for entity_id, positions in entity_positions.items():
        for x, y in positions:
            if 0 <= x < grid_size and 0 <= y < grid_size:
                heatmap_data[y, x] += 1

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=list(range(grid_size)),
        y=list(range(grid_size)),
        colorscale="Viridis"
    ))

    fig_heatmap.update_layout(
        title="Agent Position Heatmap",
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate"
    )

    # 2. Entity Positions Graph
    fig_positions = go.Figure()
    for entity_id, positions in entity_positions.items():
        if positions:
            x_coords, y_coords = zip(*positions)
            fig_positions.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="lines+markers",
                    name=f"Entity {entity_id}",
                )
            )
    fig_positions.update_layout(
        title="Entity Positions",
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        xaxis=dict(range=[0, grid_size]),
        yaxis=dict(range=[0, grid_size]),
        showlegend=True
    )

    # 3. Message Word Cloud
    if words_in_messages:
        word_counts = {}
        for word in words_in_messages:
            word_counts[word] = word_counts.get(word, 0) + 1

        fig_wordcloud = px.imshow(generate_wordcloud(word_counts))

        fig_wordcloud.update_layout(title_text='Word Cloud of Messages')
        fig_wordcloud.update_xaxes(showticklabels=False)
        fig_wordcloud.update_yaxes(showticklabels=False)
    else:
        fig_wordcloud = go.Figure().update_layout(title_text='Word Cloud of Messages (No Data)')

    # 4. Message Counts Graph
    fig_message_counts = go.Figure(
        data=[go.Scatter(x=list(timestamps), y=list(message_counts), mode="lines+markers")]
    )
    fig_message_counts.update_layout(
        title="Number of Messages Over Time",
        xaxis_title="Time",
        yaxis_title="Message Count",
    )

    # 5. HTTP Requests Graph
    if http_requests:
        status_counts = {}
        for req in http_requests:
            status = req['status']
            status_counts[status] = status_counts.get(status, 0) + 1

        fig_http = go.Figure(data=[go.Bar(
            x=list(status_counts.keys()),
            y=list(status_counts.values()),
        )])
        fig_http.update_layout(
            title="HTTP Request Status Codes",
            xaxis_title="Status Code",
            yaxis_title="Count",
        )
    else:
        fig_http = go.Figure().update_layout(title_text='HTTP Request Status Codes (No Data)')

    return fig_heatmap, fig_positions, fig_wordcloud, fig_message_counts, fig_http

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)