from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class PromptInjectionObservation(BaseModel):
    message: str = Field(description="The text/conversation to analyze for prompt injection")
    context: Optional[List[str]] = Field(
        default=None,
        description="Conversation history turns (for multi-turn hard task). Each entry is 'role: content'.",
    )
    task_id: str = Field(description="Task identifier: 'easy' | 'medium' | 'hard'")
    step: int = Field(description="Current step index (0-based)")
    total_steps: int = Field(description="Total number of steps in this task episode")
    instruction: str = Field(
        description="Instructions for the agent on what to do with this observation"
    )


class PromptInjectionAction(BaseModel):
    classification: Literal["injection", "benign"] = Field(
        description="Whether the message is a prompt injection attack or benign"
    )
    attack_type: Optional[
        Literal["direct", "indirect", "roleplay", "system_override", "social_engineering"]
    ] = Field(
        default=None,
        description="Type of attack if classified as injection. One of: direct, indirect, roleplay, system_override, social_engineering",
    )
    explanation: str = Field(
        description="Detailed explanation of why this is or is not a prompt injection attack. For injections, identify the specific attack vector."
    )
    severity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Severity of the attack from 0.0 (low) to 1.0 (critical), if classified as injection",
    )


class StepResult(BaseModel):
    observation: Optional[PromptInjectionObservation] = Field(
        default=None,
        description="Next observation (None if episode is done)",
    )
    reward: float = Field(description="Reward for the action taken (0.0-1.0)")
    done: bool = Field(description="Whether the episode has ended")
    info: Dict = Field(default_factory=dict, description="Extra info including grading breakdown")


class EnvState(BaseModel):
    task_id: str
    step: int
    total_steps: int
    episode_rewards: List[float]
    done: bool
    current_score: float = Field(description="Running average score for this episode")


class ResetRequest(BaseModel):
    task_id: Literal["easy", "medium", "hard"] = Field(
        default="easy",
        description="Which task to start. One of: easy, medium, hard",
    )


class TaskInfo(BaseModel):
    id: str
    name: str
    difficulty: str
    description: str
    num_steps: int
