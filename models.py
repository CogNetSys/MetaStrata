from pydantic import BaseModel, Field
from typing import List, Dict


class Entity(BaseModel):
    id: int
    name: str
    x: int
    y: int
    memory: str = ""

class SimulationSettings(BaseModel):
    grid_size: int
    num_entities: int
    max_steps: int
    chebyshev_distance: int
    llm_model: str
    llm_max_tokens: int
    llm_temperature: float
    request_delay: float
    max_concurrent_requests: int

class PromptSettings(BaseModel):
    message_generation_prompt: str
    memory_generation_prompt: str
    movement_generation_prompt: str

class BatchMessage(BaseModel):
    entity_id: int
    message: str

class BatchMessagesPayload(BaseModel):
    messages: List[BatchMessage]

# models.py
class World(BaseModel):
    world_id: int
    grid_size: int
    num_agents: int
    tasks: List[Dict]
    resources: Dict
    agents: List[Dict]  # Ensure this is defined

    def summarize_state(self) -> List[float]:
        """
        Generate a normalized state summary vector for this world.
        """
        task_progress = [task["progress"] for task in self.tasks]
        avg_task_progress = sum(task_progress) / len(task_progress) if task_progress else 0.0
        completed_tasks = sum(1 for p in task_progress if p >= 1.0) / len(task_progress) if task_progress else 0.0

        agent_positions = [(agent["x"], agent["y"]) for agent in self.agents]
        mean_x = sum(pos[0] for pos in agent_positions) / len(agent_positions) if agent_positions else 0.0
        mean_y = sum(pos[1] for pos in agent_positions) / len(agent_positions) if agent_positions else 0.0

        resource_consumed_ratio = self.resources["consumed"] / self.resources["total"] if self.resources["total"] else 0.0

        return [
            avg_task_progress,
            completed_tasks,
            mean_x / self.grid_size,
            mean_y / self.grid_size,
            resource_consumed_ratio,
        ]
