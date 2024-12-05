from fastapi import APIRouter, HTTPException
from config import PromptSettings, add_log
from config import (
    DEFAULT_MESSAGE_GENERATION_PROMPT,
    DEFAULT_MEMORY_GENERATION_PROMPT,
    DEFAULT_MOVEMENT_GENERATION_PROMPT
)

router = APIRouter()

@router.get("/", response_model=PromptSettings, tags=["Prompts"])
async def get_prompts():
    try:
        add_log("Fetching current prompt templates.")
        return PromptSettings(
            message_generation_prompt=DEFAULT_MESSAGE_GENERATION_PROMPT,
            memory_generation_prompt=DEFAULT_MEMORY_GENERATION_PROMPT,
            movement_generation_prompt=DEFAULT_MOVEMENT_GENERATION_PROMPT
        )
    except Exception as e:
        add_log(f"Error fetching prompt templates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/", response_model=PromptSettings, tags=["Prompts"])
async def set_prompts(prompts: PromptSettings):
    try:
        add_log("Updating prompt templates.")
        global DEFAULT_MESSAGE_GENERATION_PROMPT, DEFAULT_MEMORY_GENERATION_PROMPT, DEFAULT_MOVEMENT_GENERATION_PROMPT

        DEFAULT_MESSAGE_GENERATION_PROMPT = prompts.message_generation_prompt
        DEFAULT_MEMORY_GENERATION_PROMPT = prompts.memory_generation_prompt
        DEFAULT_MOVEMENT_GENERATION_PROMPT = prompts.movement_generation_prompt
        return prompts
    except Exception as e:
        add_log(f"Error updating prompt templates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
