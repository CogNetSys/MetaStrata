from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from utils import add_log, logger
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict
from world import World
from schemas import Feedback
from utils import add_log, logger
from fastapi import Depends
from fastapi import FastAPI
from core.app import app

# Router for MTNN endpoints
router = APIRouter()

# Pydantic model for summary submission
class SummarySubmission(BaseModel):
    world_id: int
    summary: List[float]

class FeedbackSubmission(BaseModel):
    world_id: int
    feedback: Feedback

def get_app_state(app: FastAPI = Depends()):
    return app.state

@router.post("/submit_summary", tags=["mTNN"])
async def submit_summary(data: SummarySubmission):
    """
    Endpoint to accept world state summaries for processing by the mTNN.
    
    Args:
        data (SummarySubmission): The summary payload containing world_id and summary vector.

    Returns:
        dict: A response confirming the submission.
    """
    try:
        logger.info(f"Received summary for World {data.world_id}: {data.summary}")
        add_log(f"Summary received for World {data.world_id}: {data.summary}")
        
        # TODO: Process the summary as needed for mTNN.
        # For example, store in a database or trigger higher-level processing.
        
        return {"status": "success", "world_id": data.world_id, "summary": data.summary}
    except Exception as e:
        error_message = f"Error processing summary for World {data.world_id}: {str(e)}"
        logger.error(error_message)
        raise HTTPException(status_code=500, detail=error_message)
    
@router.post("/feedback", tags=["mTNN"])
async def receive_feedback(feedback_submission: FeedbackSubmission):
    """
    Endpoint to receive feedback from the mTNN and apply it to the specified world.
    
    Args:
        feedback_submission (FeedbackSubmission): Contains the world_id and feedback vector.
    
    Returns:
        dict: Confirmation of feedback receipt and application.
    """
    world_id = feedback_submission.world_id
    feedback = feedback_submission.feedback.dict()

    # Find the corresponding world
    world = next((w for w in app.state.worlds if w.world_id == world_id), None)
    if not world:
        error_message = f"World with ID {world_id} not found."
        logger.error(error_message)
        add_log(error_message)
        raise HTTPException(status_code=404, detail=error_message)

    try:
        # Apply feedback to the world
        world.process_feedback(feedback)
        add_log(f"Feedback applied to World {world_id}: {feedback}")
        return {"status": "success", "world_id": world_id, "feedback": feedback}
    except Exception as e:
        error_message = f"Error applying feedback to World {world_id}: {str(e)}"
        logger.error(error_message)
        add_log(error_message)
        raise HTTPException(status_code=500, detail=error_message)