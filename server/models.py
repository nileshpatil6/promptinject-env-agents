from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

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


class ToolObservation(BaseModel):
    """Observation for indirect_tool task — attack arrives via a tool output, not the user."""

    user_query: str = Field(description="The original user request that triggered the tool call")
    tool_name: Literal["read_email", "browse_web", "read_file", "read_calendar", "read_slack"] = Field(
        description="The tool that was called and returned potentially injected content"
    )
    tool_output: str = Field(description="Content returned by the tool — may contain an injection attack")
    task_id: str = Field(default="indirect_tool", description="Task identifier")
    step: int = Field(description="Current step index (0-based)")
    total_steps: int = Field(description="Total number of steps in this task episode")
    instruction: str = Field(description="Instructions for the agent")


class PipelineObservation(BaseModel):
    """Observation for pipeline task — agent monitors a multi-stage workflow for injection propagation."""

    stage: int = Field(description="Current pipeline stage (1-4): 1=read, 2=summarize, 3=plan, 4=execute")
    stage_name: str = Field(description="Name of the current pipeline stage")
    content: str = Field(description="Content at this pipeline stage — may contain propagated injection")
    scenario_id: str = Field(description="Scenario identifier")
    task_id: str = Field(default="pipeline", description="Task identifier")
    step: int = Field(description="Current step index (0-based)")
    total_steps: int = Field(description="Total number of steps in this task episode")
    instruction: str = Field(description="Instructions for the agent")


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
    injection_vector: Optional[Literal["user_message", "tool_output", "none"]] = Field(
        default=None,
        description="Where the attack originates: 'user_message' (direct), 'tool_output' (indirect via tool), or 'none' (benign)",
    )


class StepResult(BaseModel):
    observation: Optional[Any] = Field(
        default=None,
        description="Next observation (None if episode is done). Type varies by task: PromptInjectionObservation, ToolObservation, or PipelineObservation.",
    )
    reward: float = Field(description="Reward for the action taken (range varies by task)")
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
    task_id: Literal["easy", "medium", "hard", "indirect_tool", "pipeline"] = Field(
        default="easy",
        description="Which task to start. One of: easy, medium, hard, indirect_tool, pipeline",
    )


class TaskInfo(BaseModel):
    id: str
    name: str
    difficulty: str
    description: str
    num_steps: int


class EvolveRequest(BaseModel):
    failed_cases: List[Dict[str, Any]] = Field(
        description="List of cases the agent failed — used to generate harder variants targeting the same blind spots"
    )
    n_variants: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of new attack variants to generate",
    )
    target_difficulty: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Desired difficulty of generated variants (0.0 = easy, 1.0 = very hard)",
    )


class EvolveResponse(BaseModel):
    variants: List[Dict[str, Any]] = Field(
        description="Generated attack variants targeting the detected blind spots"
    )
    generation: int = Field(description="Generation number (increments with each /evolve call)")
    total_dynamic_attacks: int = Field(description="Total attacks in the dynamic pool after this call")
