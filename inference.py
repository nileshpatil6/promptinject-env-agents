"""
Inference script for the Prompt Injection Detector OpenEnv environment.
Uses fine-tuned Gemma 3 1B as the primary classifier.

Environment variables:
  ADAPTER_PATH  - Path to gemma3-1b-lora adapter directory (required)
  HF_TOKEN      - HuggingFace token for loading base model
  ENV_BASE_URL  - Override environment server URL (default: localhost:7860)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import httpx

ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "dataset/gemma3-1b-lora")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

BENCHMARK = "prompt-injection-detector"
TASKS = ["easy", "medium", "hard", "indirect_tool", "pipeline"]
MAX_STEPS_PER_TASK = 15
SUCCESS_SCORE_THRESHOLD = 0.5
MODEL_NAME = "gemma3-1b-finetuned"

_detector = None

def _get_detector():
    global _detector
    if _detector is None:
        from server.gemma3_1b_detector import Gemma3_1BDetector
        _detector = Gemma3_1BDetector(ADAPTER_PATH, hf_token=HF_TOKEN)
    return _detector


# ---------------------------------------------------------------------------
# Structured logging (mandatory format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    try:
        data = json.loads(action)
        cls = data.get("classification", "unknown")
        atype = data.get("attack_type") or "null"
        sev = data.get("severity")
        sev_str = f"{sev:.2f}" if sev is not None else "null"
        action_token = f"classification={cls},attack_type={atype},severity={sev_str}"
    except Exception:
        action_token = action.replace("\n", " ").replace("\r", "").replace(" ", "_").strip()
    done_str = "true" if done else "false"
    error_str = error if error else "null"
    print(
        f"[STEP] step={step} action={action_token} reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# Gemma classifier
# ---------------------------------------------------------------------------

def get_model_action(observation: Dict[str, Any]) -> str:
    task_id = observation.get("task_id", "easy")
    detector = _get_detector()

    if task_id == "indirect_tool":
        tool_name = observation.get("tool_name", "unknown_tool")
        tool_output = observation.get("tool_output", "")
        user_query = observation.get("user_query", "")
        raw_text = f"User request: {user_query}\nTool: {tool_name}\nTool output:\n{tool_output}"

    elif task_id == "pipeline":
        stage = observation.get("stage", 1)
        stage_name = observation.get("stage_name", "")
        content = observation.get("content", "")
        raw_text = f"Pipeline stage {stage} ({stage_name}):\n{content}"

    elif task_id == "hard":
        context = observation.get("context") or []
        message = observation.get("message", "")
        numbered = "\n".join(f"Turn {i+1}: {msg}" for i, msg in enumerate(context))
        last_turn = len(context) + 1
        raw_text = f"{numbered}\nTurn {last_turn}: {message}"

    else:
        message = observation.get("message", "")
        context = observation.get("context")
        if context:
            raw_text = "\n".join(context) + "\n" + message
        else:
            raw_text = message

    try:
        result = detector.classify(raw_text, task_id=task_id)

        # Post-process: enforce task-specific rules the fine-tuned model may miss
        if task_id == "indirect_tool" and result.get("classification") == "injection":
            result["attack_type"] = "indirect"
            result["injection_vector"] = "tool_output"

        return json.dumps(result)
    except Exception as exc:
        print(f"[DEBUG] Gemma classify failed: {exc}", flush=True)
        return json.dumps({
            "classification": "benign",
            "attack_type": None,
            "explanation": "Classifier failed",
            "severity": None,
            "injection_vector": None,
        })


# ---------------------------------------------------------------------------
# Environment HTTP client
# ---------------------------------------------------------------------------

def env_reset(http: httpx.Client, task_id: str) -> Dict[str, Any]:
    resp = http.post("/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _sanitize_json(text: str) -> str:
    return re.sub(r"\\(?![\"\\\/bfnrtu])", "", text)


def env_step(http: httpx.Client, action_json: str) -> Dict[str, Any]:
    _default = {
        "classification": "benign",
        "attack_type": None,
        "explanation": "Failed to parse model response",
        "severity": None,
        "injection_vector": None,
    }
    action_data = None
    for candidate in [action_json, _sanitize_json(action_json)]:
        try:
            action_data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', candidate, re.DOTALL)
            if match:
                try:
                    action_data = json.loads(match.group())
                    break
                except json.JSONDecodeError:
                    pass
    if action_data is None:
        action_data = _default

    resp = http.post("/step", json=action_data, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

def run_task(http: httpx.Client, task_id: str) -> Dict[str, Any]:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        result = env_reset(http, task_id)
        observation = result.get("observation")
        done = result.get("done", False)

        step = 0
        while observation and not done and step < MAX_STEPS_PER_TASK:
            step += 1

            action_str = get_model_action(observation)

            try:
                result = env_step(http, action_str)
            except Exception as e:
                log_step(step=step, action=action_str, reward=0.0, done=True, error=str(e))
                break

            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            observation = result.get("observation")
            error = result.get("info", {}).get("error")

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

        if rewards:
            score = sum(rewards) / len(rewards)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task '{task_id}' failed: {e}", flush=True)
        success = False

    log_end(success=success, steps=steps_taken, score=round(score, 4), rewards=rewards)
    return {"task_id": task_id, "score": round(score, 4), "success": success, "steps": steps_taken}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"[DEBUG] Model=Gemma3-1B adapter={ADAPTER_PATH}", flush=True)
    print(f"[DEBUG] ENV_BASE_URL={ENV_BASE_URL}", flush=True)

    print("[DEBUG] Loading Gemma 3 1B detector...", flush=True)
    _get_detector()
    print("[DEBUG] Detector ready.", flush=True)

    all_results = []

    with httpx.Client(base_url=ENV_BASE_URL) as http:
        try:
            health = http.get("/health", timeout=10).json()
            print(f"[DEBUG] Environment healthy: {health}", flush=True)
        except Exception as e:
            print(f"[DEBUG] Environment not reachable at {ENV_BASE_URL}: {e}", flush=True)
            sys.exit(1)

        for task_id in TASKS:
            print(f"\n[DEBUG] === Running task: {task_id} ===", flush=True)
            result = run_task(http, task_id)
            all_results.append(result)
            time.sleep(0.5)

    print("\n[DEBUG] === INFERENCE COMPLETE ===", flush=True)
    overall_score = sum(r["score"] for r in all_results) / len(all_results)
    for r in all_results:
        status = "PASS" if r["success"] else "FAIL"
        print(f"[DEBUG] [{status}] Task={r['task_id']} Score={r['score']:.4f} Steps={r['steps']}", flush=True)
    print(f"[DEBUG] Overall average score: {overall_score:.4f}", flush=True)


if __name__ == "__main__":
    main()
