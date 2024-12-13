from pydantic import BaseModel
from typing import Dict, Optional

class Feedback(BaseModel):
    task_priorities: Optional[Dict[int, float]] = {}
    agent_behaviors: Optional[Dict[str, float]] = {}
    resource_allocation: Optional[Dict[str, int]] = {}
    environment_changes: Optional[Dict[str, float]] = {}

class FeedbackSubmission(BaseModel):
    world_id: int
    feedback: Feedback
