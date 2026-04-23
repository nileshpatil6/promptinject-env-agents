from __future__ import annotations

import json
import os
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import attacker as attacker_mod
from .env import PromptInjectionEnv
from .models import (
    EnvState,
    EvolveRequest,
    EvolveResponse,
    PromptInjectionAction,
    ResetRequest,
    StepResult,
    TaskInfo,
)
from .pipeline_env import PipelineEnv

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
DYNAMIC_ATTACKS_PATH = os.path.join(os.path.dirname(__file__), "data", "dynamic_attacks.json")

app = FastAPI(
    title="Prompt Injection Detector — OpenEnv",
    description=(
        "A real-world AI safety environment modeling prompt injection across five threat surfaces: "
        "direct attacks, subtle buried injections, multi-turn social engineering, "
        "indirect tool injection, and multi-stage pipeline propagation. "
        "Includes an adversarial self-play loop (/evolve) that generates harder variants from failure cases."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_direct_env = PromptInjectionEnv()
_pipeline_env = PipelineEnv()
_active_env: str = "direct"  # "direct" | "pipeline"
_evolve_generation: int = 0


def _get_active_env():
    if _active_env == "pipeline":
        return _pipeline_env
    return _direct_env


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "environment": "prompt-injection-detector", "version": "2.0.0"}


@app.get("/metadata")
def metadata() -> dict:
    return {
        "name": "prompt-injection-detector",
        "description": (
            "Five-task OpenEnv environment for prompt injection detection and defense. "
            "Tasks: direct (easy/medium/hard), indirect tool injection, and multi-stage pipeline defense. "
            "Includes adversarial /evolve endpoint for self-improving benchmark generation."
        ),
        "version": "2.0.0",
        "tasks": ["easy", "medium", "hard", "indirect_tool", "pipeline"],
    }


@app.get("/schema")
def schema() -> dict:
    return {
        "action": {
            "type": "object",
            "properties": {
                "classification": {"type": "string", "enum": ["injection", "benign"]},
                "attack_type": {
                    "type": "string",
                    "enum": ["direct", "indirect", "roleplay", "system_override", "social_engineering"],
                    "nullable": True,
                },
                "explanation": {"type": "string"},
                "severity": {"type": "number", "minimum": 0.0, "maximum": 1.0, "nullable": True},
                "injection_vector": {
                    "type": "string",
                    "enum": ["user_message", "tool_output", "none"],
                    "nullable": True,
                    "description": "Used in indirect_tool task: where the attack originated",
                },
            },
            "required": ["classification", "explanation"],
        },
        "observations": {
            "direct_tasks": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "context": {"type": "array", "items": {"type": "string"}, "nullable": True},
                    "task_id": {"type": "string"},
                    "step": {"type": "integer"},
                    "total_steps": {"type": "integer"},
                    "instruction": {"type": "string"},
                },
            },
            "indirect_tool": {
                "type": "object",
                "properties": {
                    "user_query": {"type": "string"},
                    "tool_name": {"type": "string", "enum": ["read_email", "browse_web", "read_file", "read_calendar", "read_slack"]},
                    "tool_output": {"type": "string"},
                    "task_id": {"type": "string"},
                    "step": {"type": "integer"},
                    "total_steps": {"type": "integer"},
                    "instruction": {"type": "string"},
                },
            },
            "pipeline": {
                "type": "object",
                "properties": {
                    "stage": {"type": "integer", "minimum": 1, "maximum": 4},
                    "stage_name": {"type": "string"},
                    "content": {"type": "string"},
                    "scenario_id": {"type": "string"},
                    "task_id": {"type": "string"},
                    "step": {"type": "integer"},
                    "total_steps": {"type": "integer"},
                    "instruction": {"type": "string"},
                },
            },
        },
    }


@app.post("/mcp")
def mcp(request: dict = None) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": (request or {}).get("id"),
        "result": {
            "name": "prompt-injection-detector",
            "description": "OpenEnv environment for prompt injection detection — 5 tasks, adversarial self-play",
        },
    }


@app.post("/reset", response_model=StepResult)
def reset(request: ResetRequest = None) -> StepResult:
    """
    Reset the environment and start a new episode.

    task_id options:
    - "easy"          → Direct injection detection (10 steps)
    - "medium"        → Subtle injection detection (10 steps)
    - "hard"          → Multi-turn social engineering (5 steps)
    - "indirect_tool" → Injection via tool outputs (10 steps)
    - "pipeline"      → Multi-stage pipeline defense (up to 4 stages)
    """
    global _active_env
    if request is None:
        request = ResetRequest()

    try:
        if request.task_id == "pipeline":
            _active_env = "pipeline"
            return _pipeline_env.reset()
        else:
            _active_env = "direct"
            return _direct_env.reset(task_id=request.task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResult)
def step(action: PromptInjectionAction) -> StepResult:
    """
    Submit an action and receive the next observation + reward.

    For all tasks:
    - classification: "injection" or "benign"
    - explanation: reasoning for your decision
    - attack_type (optional): attack category if injection
    - severity (optional): 0.0-1.0 if injection

    Additional for indirect_tool task:
    - injection_vector: "user_message" | "tool_output" | "none"

    For pipeline task: "injection" = flag_and_halt, "benign" = pass_through to next stage.
    """
    try:
        return _get_active_env().step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=EnvState)
def state() -> EnvState:
    return _get_active_env().state()


@app.get("/tasks", response_model=List[TaskInfo])
def list_tasks() -> List[TaskInfo]:
    direct_tasks = PromptInjectionEnv.get_task_info()
    pipeline_task = [PipelineEnv.get_task_info()]
    return direct_tasks + pipeline_task


@app.get("/tasks/{task_id}", response_model=TaskInfo)
def get_task(task_id: str) -> TaskInfo:
    if task_id == "pipeline":
        return PipelineEnv.get_task_info()
    try:
        return PromptInjectionEnv.get_task_info(task_id)[0]
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")


@app.post("/evolve", response_model=EvolveResponse)
def evolve(request: EvolveRequest) -> EvolveResponse:
    """
    Adversarial self-play: generate harder attack variants from failure cases.

    Pass failed_cases (dicts with 'message', 'agent_classification', 'true_classification')
    and receive new attack variants targeting the same blind spots.
    Generated variants are appended to the dynamic attack pool.
    """
    global _evolve_generation
    _evolve_generation += 1

    variants = attacker_mod.generate_variants(
        failed_cases=request.failed_cases,
        n_variants=request.n_variants,
        target_difficulty=request.target_difficulty,
        generation=_evolve_generation,
    )

    try:
        with open(DYNAMIC_ATTACKS_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing = []

    existing.extend(variants)

    with open(DYNAMIC_ATTACKS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    return EvolveResponse(
        variants=variants,
        generation=_evolve_generation,
        total_dynamic_attacks=len(existing),
    )


@app.get("/stats")
def stats() -> dict:
    """Return current environment statistics including dynamic attack pool size."""
    try:
        with open(DYNAMIC_ATTACKS_PATH, "r", encoding="utf-8") as f:
            dynamic = json.load(f)
        pool_size = len(dynamic)
        generations = _evolve_generation
    except Exception:
        pool_size = 0
        generations = 0

    return {
        "evolve_generation": generations,
        "dynamic_attack_pool_size": pool_size,
        "active_task": _active_env,
        "tasks_available": 5,
    }


@app.get("/")
def root() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
