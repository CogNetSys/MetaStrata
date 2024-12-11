from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from utils import add_log, logger

# Router for MTNN endpoints
router = APIRouter()

# Pydantic model for summary submission
class SummarySubmission(BaseModel):
    world_id: int
    summary: List[float]

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
