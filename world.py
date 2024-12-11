import numpy as np
from typing import List, Dict
from utils import add_log
import logging

logger = logging.getLogger("simulation_app")

class World:
    def __init__(self, world_id: int, grid_size: int, num_agents: int, tasks: List[Dict], resources: Dict):
        self.world_id = world_id
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.agents = self.initialize_agents(num_agents)
        self.tasks = tasks
        self.resources = resources
        self.communications = []  # List to store communication activities
        self.emergent_metrics = {}

    def initialize_agents(self, num_agents: int) -> List[Dict]:
        agents = []
        for i in range(num_agents):
            agent = {
                "id": i,
                "x": np.random.randint(0, self.grid_size),
                "y": np.random.randint(0, self.grid_size),
                "memory": "",
                "cooperation_score": np.random.rand(),
                "conflict_score": np.random.rand()
            }
            agents.append(agent)
        logger.info(f"Initialized {num_agents} agents for World {self.world_id}.")
        return agents

    def summarize_state(self) -> List[float]:
        try:
            # Task Metrics
            task_progress = [task.get("progress", 0) for task in self.tasks]
            avg_task_progress = np.mean(task_progress) if task_progress else 0
            fraction_completed_tasks = np.sum([1 for p in task_progress if p >= 1.0]) / len(task_progress) if task_progress else 0
            variance_task_durations = np.var([task.get("duration", 0) for task in self.tasks]) if self.tasks else 0

            # Agent Metrics
            agent_x = [agent["x"] for agent in self.agents]
            agent_y = [agent["y"] for agent in self.agents]
            mean_x = np.mean(agent_x) if agent_x else 0
            mean_y = np.mean(agent_y) if agent_y else 0
            var_x = np.var(agent_x) if agent_x else 0
            var_y = np.var(agent_y) if agent_y else 0
            avg_cooperation = np.mean([agent["cooperation_score"] for agent in self.agents]) if self.agents else 0
            avg_conflict = np.mean([agent["conflict_score"] for agent in self.agents]) if self.agents else 0

            # Communication Metrics
            total_messages = len(self.communications)
            avg_messages_per_agent = total_messages / self.num_agents if self.num_agents else 0

            # Resource Metrics
            total_resources = self.resources.get("total", 0)
            consumed_resources = self.resources.get("consumed", 0)
            percentage_resources_consumed = consumed_resources / total_resources if total_resources else 0
            resource_distribution = self.resources.get("distribution", {})
            # For simplicity, let's take the mean distribution if it's a list
            mean_resource_distribution = np.mean(list(resource_distribution.values())) if resource_distribution else 0

            # Emergent Metrics
            cooperation_ratio = avg_cooperation / (avg_cooperation + avg_conflict) if (avg_cooperation + avg_conflict) else 0
            entropy = self.calculate_entropy()

            # Normalize metrics between 0 and 1 where applicable
            normalized_metrics = [
                avg_task_progress,  # Assuming progress is already between 0 and 1
                fraction_completed_tasks,
                variance_task_durations / 100,  # Assuming max variance is 100
                mean_x / self.grid_size,
                mean_y / self.grid_size,
                var_x / (self.grid_size ** 2),
                var_y / (self.grid_size ** 2),
                avg_cooperation,
                avg_conflict,
                avg_messages_per_agent / 100,  # Assuming max messages per agent is 100
                percentage_resources_consumed,
                mean_resource_distribution / 100,  # Assuming max distribution value is 100
                cooperation_ratio,
                entropy / np.log2(len(self.agents)) if self.agents else 0  # Normalize entropy
            ]

            # Ensure fixed-size vector
            summary_vector = normalized_metrics[:14]  # Adjust based on the number of metrics
            logger.debug(f"World {self.world_id} summary vector: {summary_vector}")
            return summary_vector

        except Exception as e:
            logger.error(f"Error summarizing state for World {self.world_id}: {e}")
            add_log(f"Error summarizing state for World {self.world_id}: {e}")
            # Return a default vector with zeros in case of error
            return [0.0] * 14

    def calculate_entropy(self) -> float:
        try:
            # Example: Calculate Shannon entropy for agent positions
            positions = [(agent["x"], agent["y"]) for agent in self.agents]
            unique_positions, counts = np.unique(positions, axis=0, return_counts=True)
            probabilities = counts / counts.sum()
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Add small value to prevent log(0)
            return entropy
        except Exception as e:
            logger.error(f"Error calculating entropy for World {self.world_id}: {e}")
            return 0.0

    # Additional methods to update agents, tasks, resources, and communications can be added here
