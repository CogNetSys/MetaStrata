# /endpoints/visualizations.py


import io
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from endpoints.database import redis, supabase
from config import GRID_SIZE
from utils import add_log, LOG_QUEUE, logger

grid_size = GRID_SIZE

router = APIRouter()

# Entity Grid
@router.get("/grid", tags=["Visualization"])
async def generate_grid_visualization():
    """
    Generate a grid visualization displaying entities' numbers within their respective cells.
    """
    try:
        # Fetch all entities from Redis
        entity_keys = await redis.keys("entity:*")
        if not entity_keys:
            raise HTTPException(status_code=404, detail="No entities found.")

        # Get grid size from configuration
        grid_size = GRID_SIZE  # Assume GRID_SIZE is defined in your settings

        # Create a blank grid
        grid = np.full((grid_size, grid_size), "", dtype=object)

        # Place entities on the grid
        for key in entity_keys:
            entity_data = await redis.hgetall(key)
            x, y, entity_id = int(entity_data["x"]), int(entity_data["y"]), entity_data["id"]
            grid[y][x] = str(entity_id)  # Place the entity number at the position

        # Plot the grid
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks(np.arange(0, grid_size, 1))
        ax.set_yticks(np.arange(0, grid_size, 1))
        ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
        ax.tick_params(which="major", bottom=False, left=False, labelbottom=False, labelleft=False)

        # Annotate with entity numbers
        for y in range(grid_size):
            for x in range(grid_size):
                if grid[y][x]:
                    ax.text(x + 0.5, grid_size - y - 0.5, grid[y][x], ha="center", va="center", color="blue")

        # Save the grid to a BytesIO stream
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        # Return the image as a StreamingResponse
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        error_message = f"Error generating grid visualization: {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)

# Heatmap visualization
@router.get("/visualization/heatmap", tags=["Visualization"])
async def agent_location_heatmap():
    """
    Generate a heatmap of agent locations.
    """
    try:
        # Fetch all entities from Redis
        entity_keys = await redis.keys("entity:*")
        if not entity_keys:
            raise HTTPException(status_code=404, detail="No entities found.")

        # Initialize the grid
        heatmap = np.zeros((GRID_SIZE, GRID_SIZE))

        # Increment the grid based on agent locations
        for key in entity_keys:
            entity_data = await redis.hgetall(key)
            x, y = int(entity_data["x"]), int(entity_data["y"])
            heatmap[y][x] += 1

        # Create a heatmap visualization
        fig, ax = plt.subplots(figsize=(8, 8))
        cax = ax.matshow(heatmap, cmap="viridis", origin="lower")
        plt.colorbar(cax)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        error_message = f"Error generating heatmap: {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)


# Trajectory visualization
@router.get("/visualization/trajectory", tags=["Visualization"])
async def trajectory_visualization():
    try:
        # Retrieve entity keys
        entity_keys = await redis.keys("entity:*")
        if not entity_keys:
            return {"error": "No entities found for trajectory visualization."}

        # Fetch entity positions
        trajectories = []
        for key in entity_keys:
            entity_data = await redis.hgetall(key)
            if entity_data:
                trajectories.append((int(entity_data["x"]), int(entity_data["y"])))

        # Generate plot
        plt.figure(figsize=(10, 10))
        x_vals, y_vals = zip(*trajectories) if trajectories else ([], [])
        plt.plot(x_vals, y_vals, marker='o')
        plt.title("Agent Trajectories")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)

        # Return the plot as an image
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        return {"message": f"Custom Error: Error generating trajectory visualization: {str(e)}"}

# Interaction network graph visualization
@router.get("/visualization/network", tags=["Visualization"])
async def interaction_network_graph():
    """
    Generate a graph visualization of agent interactions.
    """
    try:
        # Fetch interactions from Redis
        interactions = []  # Replace with logic to fetch agent interactions
        graph = nx.Graph()

        for interaction in interactions:
            entity_a, entity_b = interaction["from"], interaction["to"]
            graph.add_edge(entity_a, entity_b)

        # Draw the graph
        fig, ax = plt.subplots(figsize=(10, 10))
        nx.draw_networkx(graph, ax=ax, node_size=700, font_size=10)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        error_message = f"Error generating interaction graph: {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)

# Behavior over time visualization
@router.get("/visualization/behavior", tags=["Visualization"])
async def behavior_over_time():
    """
    Generate a time-series visualization of agent behavior.
    """
    try:
        # Fetch behavior data from Redis
        behavior_data = {}  # Replace with logic to fetch behavior data

        # Prepare the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for agent_id, data in behavior_data.items():
            times, values = zip(*data)
            ax.plot(times, values, label=f"Agent {agent_id}")

        ax.set_title("Behavior Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Behavior Metric")
        ax.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        error_message = f"Error generating behavior plot: {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)

# Task allocation chart visualization
@router.get("/visualization/task_allocation", tags=["Visualization"])
async def task_allocation_chart():
    try:
        # Retrieve task allocation data
        task_data = await redis.hgetall("tasks")  # Replace with appropriate Redis call
        if not task_data:
            return {"message": "No task allocation data found."}

        # Verify task_data is a dictionary
        if isinstance(task_data, list):
            raise ValueError("Task data is a list; expected a dictionary.")

        # Prepare data for visualization
        tasks = list(task_data.keys())
        allocations = [len(eval(entities)) for entities in task_data.values()]  # Example: Eval if data is stored as strings

        # Generate bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(tasks, allocations, color="skyblue")
        plt.title("Task Allocation Chart")
        plt.xlabel("Tasks")
        plt.ylabel("Number of Assigned Entities")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Return the chart as an image
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        return {"message": f"Custom Error: Error generating task allocation chart: {str(e)}"}

# System metrics over time visualization
@router.get("/visualization/system_metrics", tags=["Visualization"])
async def system_metrics_over_time():
    """
    Generate a time-series visualization of system-wide metrics.
    """
    try:
        # Fetch system metrics data
        metrics = {}  # Replace with logic to fetch system metrics

        # Prepare the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for metric_name, data in metrics.items():
            times, values = zip(*data)
            ax.plot(times, values, label=metric_name)

        ax.set_title("System Metrics Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Metric Value")
        ax.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        error_message = f"Error generating system metrics plot: {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)
