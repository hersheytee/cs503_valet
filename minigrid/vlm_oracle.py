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

_DIR_WORD = {"→": "RIGHT", "↓": "DOWN", "←": "LEFT", "↑": "UP"}

_SYSTEM_PROMPT = """\
You are an expert navigation assistant for a grid-world agent.
Your role is to analyze the agent's current visual observation and recommend \
the single best action to make progress toward the mission goal.\
"""

_VALID_ACTIONS = {0, 1, 2, 3, 5}   # turn_left, turn_right, forward, pickup, toggle

MODES = ("baseline", "cot", "thinking")
_MAX_TOKENS = {"baseline": 8, "cot": 300, "thinking": 600}


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


# ── Phase detection ───────────────────────────────────────────────────────────

def _detect_phase(env_unwrapped) -> str:
    """
    Mirrors _detect_phase from run_benchmark.py, but operates on a live env.
      "navigate"  — Empty env, or DoorKey with door already open
      "find_key"  — DoorKey env, key not yet carried
      "open_door" — DoorKey env, key in hand, door still closed
    """
    grid = env_unwrapped.grid
    W, H = env_unwrapped.width, env_unwrapped.height

    door_cell = None
    for x in range(W):
        for y in range(H):
            cell = grid.get(x, y)
            if cell is not None and cell.type == 'door':
                door_cell = cell
                break
        if door_cell is not None:
            break

    if door_cell is None:
        return "navigate"

    carrying = env_unwrapped.carrying
    has_key  = carrying is not None and carrying.type == 'key'

    if not has_key:
        return "find_key"
    if not door_cell.is_open:
        return "open_door"
    return "navigate"


# ── State extraction from env ─────────────────────────────────────────────────

def _extract_state(env_unwrapped) -> dict:
    """Pulls agent state metadata from a raw MiniGrid env."""
    dir_idx      = int(env_unwrapped.agent_dir)
    dir_arrow    = _DIR_ARROW.get(dir_idx, "?")
    carrying     = env_unwrapped.carrying
    carrying_str = f"{carrying.color} {carrying.type}" if carrying is not None else None
    mission      = getattr(env_unwrapped, "mission", "reach the goal")
    return {
        "agent_dir_str":  dir_arrow,
        "agent_carrying": carrying_str,
        "mission":        mission,
        "phase":          _detect_phase(env_unwrapped),
    }


# ── Prompt builder — negative_rules variant (mirrors run_benchmark.py) ────────

def _phase_goal(phase: str, fw: str) -> str:
    if phase == "find_key":
        return (
            f"CURRENT SUB-GOAL: Pick up the YELLOW KEY — you do not have it yet.\n"
            f"  Navigate toward the key. Use action 3 (pickup) only when the key is "
            f"directly {fw} of you. Ignore the door for now."
        )
    if phase == "open_door":
        return (
            f"CURRENT SUB-GOAL: Open the DOOR — you ARE carrying the key.\n"
            f"  Navigate toward the door. Use action 5 (toggle) only when the door "
            f"is directly {fw} of you. Then walk through to the goal."
        )
    return (
        f"CURRENT SUB-GOAL: Reach the GREEN GOAL square.\n"
        f"  The path is clear (or the door is open). Navigate directly to the goal."
    )


def _cot_suffix(cot: bool, thinking: bool) -> str:
    if thinking:
        return (
            "\n\nThink step by step about what you see and what you should do. "
            "Then write your final answer as: <answer>N</answer>  (N is the action integer)."
        )
    if cot:
        return (
            "\n\nBriefly reason step by step (2-3 sentences), "
            "then give your final answer as ONE integer."
        )
    return "\n\nREPLY WITH ONE INTEGER ONLY."


def _build_user_text(state: dict, cot: bool = False, thinking: bool = False) -> str:
    """Negative-rules prompt — mirrors _v_negative from run_benchmark.py."""
    dir_str = state["agent_dir_str"]
    fw      = _DIR_WORD.get(dir_str, dir_str)
    carry   = state["agent_carrying"]
    phase   = state["phase"]

    lines = [
        f"Grid agent faces {fw}. "
        f"{'Carrying: ' + carry + '.' if carry else 'Carrying: nothing.'}  "
        f"Mission: {state['mission']}",
        "",
        _phase_goal(phase, fw),
        "",
        "Rules:",
        f"  ✗ Do NOT pickup (3) unless a KEY is DIRECTLY {fw} and you hold nothing.",
        f"  ✗ Do NOT toggle (5) unless a DOOR is DIRECTLY {fw} AND you hold the key.",
        f"  ✗ Do NOT move forward (2) into a wall or closed door.",
        f"  ✓ DO pickup the key if it is directly {fw}.",
        f"  ✓ DO toggle the door if directly {fw} and key held.",
        f"  ✓ DO move forward when path is clear and moves you closer to target.",
        f"  ✓ DO turn (0 or 1) to face your next sub-goal.",
        "",
        f"Actions: 0=turn_left  1=turn_right  2=forward({fw})  3=pickup  5=toggle",
        _cot_suffix(cot, thinking),
    ]
    return "\n".join(lines)


def _build_messages(obs_b64: str, state: dict, model_key: str,
                    cot: bool = False, thinking: bool = False) -> list[dict]:
    """Assembles chat messages in OpenAI format for the given model."""
    image_block = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{obs_b64}"},
    }
    user_text = _build_user_text(state, cot, thinking)

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

def _parse_response(raw: str, thinking: bool = False) -> Optional[int]:
    """
    Extracts the action integer from a VLM response.
    In thinking mode, looks for <answer>N</answer> first.
    Returns None if no valid action can be extracted.
    """
    if not raw:
        return None
    text = raw.strip()

    # Thinking mode: <answer>N</answer> takes priority
    if thinking:
        m = re.search(r"<answer>\s*(\d+)\s*</answer>", text, re.IGNORECASE)
        if m:
            a = int(m.group(1))
            if a in _VALID_ACTIONS:
                return a

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
    mode:              str = "baseline",
) -> tuple[int, bool]:
    """
    Queries the VLM server and returns (action, fallback_used).

    Parameters
    ----------
    obs_img           : current RGB observation (H, W, 3) uint8 numpy array
    env_unwrapped     : raw MiniGrid env (for agent_dir, carrying, mission)
    client            : OpenAI client from make_vlm_client()
    served_model_name : model ID returned by get_served_model_name()
    model_key         : key in MODELS dict (e.g. "qwen3b")
    action_history    : list of recent action names, most recent last (optional)
    fallback_action   : action to return if the VLM response cannot be parsed
    mode              : "baseline" | "cot" | "thinking"

    Returns
    -------
    (int, bool) : action index, and whether a fallback was used
    """
    if action_history is None:
        action_history = []

    cot      = (mode == "cot")
    thinking = (mode == "thinking")

    state    = _extract_state(env_unwrapped)
    obs_b64  = _obs_to_b64(obs_img)
    messages = _build_messages(obs_b64, state, model_key, cot, thinking)

    max_tokens = _MAX_TOKENS.get(mode, 8)

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
        return fallback_action, True  # (action, fallback_used)

    action = _parse_response(raw, thinking=thinking)
    if action is None:
        print(f"[vlm_oracle] Could not parse response {repr(raw)!r}, using fallback {fallback_action}.", flush=True)
        return fallback_action, True  # (action, fallback_used)

    return action, False  # (action, fallback_used)


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