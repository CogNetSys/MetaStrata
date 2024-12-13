# message.py
from typing import Optional, Dict, Any
from pydantic import BaseModel

class Message(BaseModel):
    source_world_id: int
    target_world_id: Optional[int]  # None indicates a broadcast
    message_type: str              # e.g., "state_summary", "request", "alert"
    payload: Dict[str, Any]        # Message-specific data
