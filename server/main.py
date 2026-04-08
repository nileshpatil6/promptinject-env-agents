from __future__ import annotations

from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .env import PromptInjectionEnv
from .models import (
    EnvState,
    PromptInjectionAction,
    ResetRequest,
    StepResult,
    TaskInfo,
)

app = FastAPI(
    title="Prompt Injection Detector — OpenEnv",
    description=(
        "A real-world AI safety environment where agents learn to detect prompt injection attacks. "
        "Three tasks of increasing difficulty: direct attacks, subtle buried injections, "
        "and multi-turn social engineering conversations."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global environment instance (stateful per-session)
_env = PromptInjectionEnv()


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "environment": "prompt-injection-detector", "version": "1.0.0"}


@app.post("/reset", response_model=StepResult)
def reset(request: ResetRequest = None) -> StepResult:
    """
    Reset the environment and start a new episode.

    Pass task_id to choose difficulty:
    - "easy"   → Direct injection detection (10 steps)
    - "medium" → Subtle injection detection (10 steps)
    - "hard"   → Multi-turn social engineering (3 steps)
    """
    if request is None:
        request = ResetRequest()
    try:
        return _env.reset(task_id=request.task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResult)
def step(action: PromptInjectionAction) -> StepResult:
    """
    Submit an action (classification + explanation) and receive the next observation + reward.

    The action must include:
    - classification: "injection" or "benign"
    - explanation: why you made this decision
    - attack_type (optional): "direct" | "indirect" | "roleplay" | "system_override" | "social_engineering"
    - severity (optional): 0.0-1.0 if injection
    """
    try:
        return _env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=EnvState)
def state() -> EnvState:
    """Return the current environment state."""
    return _env.state()


@app.get("/tasks", response_model=List[TaskInfo])
def list_tasks() -> List[TaskInfo]:
    """List all available tasks with their metadata."""
    return PromptInjectionEnv.get_task_info()


@app.get("/tasks/{task_id}", response_model=TaskInfo)
def get_task(task_id: str) -> TaskInfo:
    """Get metadata for a specific task."""
    try:
        return PromptInjectionEnv.get_task_info(task_id)[0]
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")


@app.get("/")
def root() -> dict:
    return {
        "name": "Prompt Injection Detector",
        "description": "OpenEnv environment for AI safety: detect prompt injection attacks",
        "tasks": ["easy", "medium", "hard"],
        "endpoints": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "tasks": "GET /tasks",
            "docs": "GET /docs",
        },
    }
