from pydantic import BaseModel, Field


class TaskContext(BaseModel):
    """
    Provides context about the task the trying to be accomplished.
    """
    
    goal: str = Field(
        description="The overall goal trying to be achieved"
    )

    intial_message: str = Field(
        description="The initial message or question the agent would be asked"
    )
