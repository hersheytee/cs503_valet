"""
VLM Oracle for Upper_Bound experiments.

Drop-in replacement for the BFS oracle: given the current RGB observation and
the raw MiniGrid env, queries a VLM served by vLLM and returns an action int.

Self-contained — no dependency on PromptBenchmark or Benchmark directories.

Public API
----------
    is_model_downloaded(model_key, cache_dir)  → bool
    start_vlm_server(model_key, cache_dir)     → subprocess.Popen
    wait_for_server(timeout)                   → bool
    stop_vlm_server(proc)
    make_vlm_client()                          → OpenAI
    get_served_model_name(client)              → str
    query_vlm(obs_img, env_unwrapped, client, served_model_name, model_key,
              action_history)                  → int
"""

import base64
import io
import re
import signal
import subprocess
import sys
import time
from typing import Optional

import numpy as np

from download_models import MODELS

# Models whose chat template breaks with a separate system message
# → system prompt merged into the user turn (mirrors prompt_builder.py)
_SIMPLE_FORMAT_MODELS = {"internvl", "llava", "smolvlm"}

VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
VLLM_URL  = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"

# MiniGrid direction index → arrow symbol
_DIR_ARROW = {0: "→", 1: "↓", 2: "←", 3: "↑"}

# Mirrors prompt_builder.py
_DIR_DESCRIPTION = {"→": "RIGHT →", "↓": "DOWN ↓", "←": "LEFT ←", "↑": "UP ↑"}
_DIR_WORD        = {"→": "RIGHT",   "↓": "DOWN",   "←": "LEFT",   "↑": "UP"}

_ACTIONS = {
    0: ("turn_left",  "Turn left  — rotate 90° counter-clockwise, stay in place"),
    1: ("turn_right", "Turn right — rotate 90° clockwise, stay in place"),
    2: ("forward",    "Move forward — move one cell in the direction you are facing"),
    3: ("pickup",     "Pick up — grab the object directly in front of you"),
    4: ("drop",       "Drop — place the object you are carrying in front of you"),
    5: ("toggle",     "Toggle — open/close the door directly in front of you"),
}

_VIEW_CONTEXT = (
    "The image shows a TOP-DOWN view of the ENTIRE grid. "
    "You can see all objects, walls, doors, keys, and the goal. "
    "The red triangle is the agent — its pointy tip shows which direction it faces. "
    "The agent's exact facing direction is stated in text below: use the text as ground truth."
)

_SYSTEM_PROMPT = """\
You are an expert navigation assistant for a grid-world agent.
Your role is to analyze the agent's current visual observation and recommend \
the single best action to make progress toward the mission goal.

You must respond with ONLY a single integer corresponding to the action number. \
No explanation, no text, no punctuation — just the integer.\
"""

_VALID_ACTIONS = {0, 1, 2, 3, 5}   # turn_left, turn_right, forward, pickup, toggle


# ── vLLM server lifecycle ─────────────────────────────────────────────────────

def start_vlm_server(model_key: str, cache_dir: str) -> subprocess.Popen:
    """Starts a vLLM OpenAI-compatible server. Model must already be downloaded."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model key '{model_key}'. Available: {list(MODELS)}")
    cfg = MODELS[model_key]

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model",                  cfg["repo_id"],
        "--dtype",                  cfg["dtype"],
        "--gpu-memory-utilization", str(cfg["gpu_mem"]),
        "--host",                   VLLM_HOST,
        "--port",                   str(VLLM_PORT),
        "--trust-remote-code",
        "--max-model-len",          "4096",
        "--enforce-eager",
        "--download-dir",           cache_dir,
    ]

    import os
    env = os.environ.copy()
    env["HF_HOME"]               = cache_dir
    env["HUGGINGFACE_HUB_CACHE"] = cache_dir
    env["TRANSFORMERS_OFFLINE"]  = "1"
    env["HF_DATASETS_OFFLINE"]   = "1"

    print(f"[vlm_oracle] Starting vLLM server for {cfg['name']} ...", flush=True)
    print(f"[vlm_oracle] CMD: {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(cmd, env=env, stdout=sys.stderr, stderr=sys.stderr)
    return proc


def wait_for_server(timeout: int = 300) -> bool:
    """Polls the vLLM /health endpoint until ready or timeout (seconds)."""
    import urllib.request

    health_url = f"http://{VLLM_HOST}:{VLLM_PORT}/health"
    t0 = time.time()
    print(f"[vlm_oracle] Waiting for server at {health_url} ...", flush=True)

    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(health_url, timeout=2)
            print(f"[vlm_oracle] Server ready ({time.time() - t0:.0f}s).", flush=True)
            return True
        except Exception:
            time.sleep(3)

    print(f"[vlm_oracle] Server did not start within {timeout}s.", flush=True)
    return False


def stop_vlm_server(proc: subprocess.Popen):
    """Gracefully terminates the vLLM server subprocess."""
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(3)
    print("[vlm_oracle] Server stopped.", flush=True)


# ── Client helpers ────────────────────────────────────────────────────────────

def make_vlm_client():
    """Returns an OpenAI client pointed at the local vLLM server."""
    from openai import OpenAI
    return OpenAI(base_url=VLLM_URL, api_key="vllm-local")


def get_served_model_name(client) -> str:
    """Fetches the model ID as reported by the running vLLM server."""
    try:
        return client.models.list().data[0].id
    except Exception:
        return ""


# ── Image encoding ────────────────────────────────────────────────────────────

def _obs_to_b64(obs: np.ndarray) -> str:
    """Encodes a (H, W, 3) uint8 numpy array as a base64 PNG string."""
    from PIL import Image
    img = Image.fromarray(obs.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── State extraction from env ─────────────────────────────────────────────────

def _extract_state(env_unwrapped) -> dict:
    """Pulls agent state metadata from a raw MiniGrid env."""
    dir_idx     = int(env_unwrapped.agent_dir)
    dir_arrow   = _DIR_ARROW.get(dir_idx, "?")
    carrying    = env_unwrapped.carrying
    carrying_str = f"{carrying.color} {carrying.type}" if carrying is not None else None
    mission     = getattr(env_unwrapped, "mission", "reach the goal")
    return {
        "agent_dir_str":   dir_arrow,
        "agent_carrying":  carrying_str,
        "mission":         mission,
    }


# ── Prompt builder (baseline — mirrors prompt_builder.py) ─────────────────────

def _describe_state(state: dict) -> str:
    dir_str  = state["agent_dir_str"]
    carrying = state["agent_carrying"]
    mission  = state["mission"]

    lines = [f"Mission: {mission}"]
    lines.append(
        f"Agent is facing: {_DIR_DESCRIPTION.get(dir_str, dir_str)} "
        f"— moving forward will move the agent {_DIR_WORD.get(dir_str, '')} in the grid."
    )
    if carrying:
        lines.append(f"Agent is currently carrying: a {carrying}")
    else:
        lines.append("Agent is not carrying anything")
    return "\n".join(lines)


def _build_action_menu(carrying: str | None, dir_str: str) -> str:
    forward_dir = _DIR_WORD.get(dir_str, "")
    lines = ["Available actions:"]
    for idx, (_, description) in _ACTIONS.items():
        if idx == 4 and carrying is None:
            continue
        if idx == 2 and forward_dir:
            lines.append(f"  {idx}: Move forward — move one cell {forward_dir} (the direction the agent is facing)")
        else:
            lines.append(f"  {idx}: {description}")
    lines.append("\nRespond with the action number only !")
    return "\n".join(lines)


def _build_user_text(state: dict) -> str:
    """Baseline prompt — identical structure to prompt_builder.py."""
    return (
        f"{_VIEW_CONTEXT}\n\n"
        f"{_describe_state(state)}\n\n"
        f"{_build_action_menu(state['agent_carrying'], state['agent_dir_str'])}"
    )


def _build_messages(obs_b64: str, state: dict, model_key: str) -> list[dict]:
    """Assembles chat messages in OpenAI format for the given model."""
    image_block = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{obs_b64}"},
    }
    user_text = _build_user_text(state)

    if model_key in _SIMPLE_FORMAT_MODELS:
        return [{
            "role": "user",
            "content": [
                image_block,
                {"type": "text", "text": _SYSTEM_PROMPT + "\n\n" + user_text},
            ],
        }]
    else:
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": [
                image_block,
                {"type": "text", "text": user_text},
            ]},
        ]


# ── Response parser ───────────────────────────────────────────────────────────

def _parse_response(raw: str) -> Optional[int]:
    """
    Extracts the action integer from a VLM response.
    Returns None if no valid action can be extracted.
    """
    if not raw:
        return None
    text = raw.strip()

    # Explicit answer keyword
    m = re.search(
        r"(?:answer|reply|action|final|choose|select)\s*[:\-=]*\s*(\d)",
        text, re.IGNORECASE,
    )
    if m:
        a = int(m.group(1))
        if a in _VALID_ACTIONS:
            return a

    # Last standalone digit not part of a numbered bullet
    candidates = re.findall(r"(?<![\.\d])(\d)(?![\.\d])", text)
    for token in reversed(candidates):
        a = int(token)
        if a in _VALID_ACTIONS:
            return a

    # Fuzzy keyword fallback
    text_lower = text.lower()
    keyword_map = {
        "turn left": 0, "turn_left": 0, "left": 0,
        "turn right": 1, "turn_right": 1, "right": 1,
        "forward": 2, "move": 2, "ahead": 2, "straight": 2,
        "pickup": 3, "pick up": 3, "pick": 3, "grab": 3, "collect": 3,
        "toggle": 5, "open": 5, "unlock": 5, "door": 5,
    }
    for keyword, action_id in sorted(keyword_map.items(), key=lambda x: -len(x[0])):
        if keyword in text_lower:
            return action_id

    return None


# ── Main entry point ──────────────────────────────────────────────────────────

def query_vlm(
    obs_img:          np.ndarray,
    env_unwrapped,
    client,
    served_model_name: str,
    model_key:         str,
    action_history:    list[str] | None = None,
    fallback_action:   int = 2,
) -> int:
    """
    Queries the VLM server and returns an action integer.

    Parameters
    ----------
    obs_img           : current RGB observation (H, W, 3) uint8 numpy array
    env_unwrapped     : raw MiniGrid env (for agent_dir, carrying, mission)
    client            : OpenAI client from make_vlm_client()
    served_model_name : model ID returned by get_served_model_name()
    model_key         : key in MODELS dict (e.g. "qwen3b")
    action_history    : list of recent action names, most recent last (optional)
    fallback_action   : action to return if the VLM response cannot be parsed

    Returns
    -------
    int : action index (0=turn_left, 1=turn_right, 2=forward, 3=pickup, 5=toggle)
    """
    if action_history is None:
        action_history = []

    state    = _extract_state(env_unwrapped)
    obs_b64  = _obs_to_b64(obs_img)
    messages = _build_messages(obs_b64, state, model_key)

    # Max tokens: CoT-style prompt needs more room; baseline needs just 1 digit
    max_tokens = 8

    try:
        response = client.chat.completions.create(
            model=served_model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=30,
        )
        raw = response.choices[0].message.content or ""
    except Exception as e:
        print(f"[vlm_oracle] VLM call failed: {e}", flush=True)
        return fallback_action

    action = _parse_response(raw)
    if action is None:
        print(f"[vlm_oracle] Could not parse response {repr(raw)!r}, using fallback {fallback_action}.", flush=True)
        return fallback_action

    return action


# ── CLI self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, os

    p = argparse.ArgumentParser(description="VLM oracle self-test")
    p.add_argument("--model",     default="qwen3b", choices=list(MODELS))
    p.add_argument("--cache_dir", default=os.environ.get("HF_HOME", "./hf_cache"))
    args = p.parse_args()

    from download_models import is_downloaded
    print(f"Model     : {MODELS[args.model]['name']}")
    print(f"Cache dir : {args.cache_dir}")
    print(f"Downloaded: {is_downloaded(MODELS[args.model]['repo_id'], args.cache_dir)}")

    print("\n--- Parse response tests ---")
    cases = [
        ("2",                   2),
        ("  3 \n",              3),
        ("turn left",           0),
        ("I recommend forward", 2),
        ("open the door",       5),
        ("xyz",                 None),
    ]
    all_ok = True
    for raw, expected in cases:
        got = _parse_response(raw)
        ok  = got == expected
        all_ok = all_ok and ok
        print(f"  {'OK' if ok else 'FAIL'}  parse({repr(raw):30s}) → {got}  (expected {expected})")
    print(f"\n{'All tests passed.' if all_ok else 'Some tests FAILED.'}")