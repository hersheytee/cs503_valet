"""
MiniGrid VLM Benchmark — Main Runner
=====================================
Supports the original dataset and the new history dataset.

New features vs. original:
  --history_images N  : pass N preceding frames alongside the current frame
  --cot               : prompt the model to reason step by step (more tokens)
  --thinking          : stronger reasoning prompt + <answer>N</answer> parsing
                        (best with WeThink; works as CoT for other models)

Prompt variants are now mission-aware (phase detection):
  navigate  — Empty env or door already open → go to goal
  find_key  — DoorKey env, key not in hand  → pick up key first
  open_door — DoorKey env, key in hand      → toggle the door

Multi-image notes (per model family):
  Qwen2.5-VL / Qwen2-VL / WeThink  → images referenced automatically by order
  InternVL 2.5 / 3                  → text must contain "Image-i: <image>" markers
  SmolVLM2                          → text must contain "Image-i: <image>" markers
  Launch vLLM with --limit-mm-per-prompt image=N  (N = history_images + 1)

Output:
  results/
    raw/
      qwen3b_global_if_then.json
      qwen3b_global_if_then_h3_cot.json   (with history + cot suffix)
      ...
    benchmark_results.csv
    benchmark_summary.json

Usage:
    python run_benchmark.py --dataset ./history_dataset/dataset.json

    # 3 history images, CoT, global view only, first 20 samples
    python run_benchmark.py --history_images 3 --cot --views global --max_samples 20

    # Thinking mode (best for WeThink)
    python run_benchmark.py --models wethink --thinking

Install:
    pip install vllm openai tqdm pandas
"""

import argparse
import base64
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils.prompt_builder import parse_response

# ── Constants ────────────────────────────────────────────────────────────────

_DIR_WORD = {"→": "RIGHT", "↓": "DOWN", "←": "LEFT", "↑": "UP"}

MODELS = {
    "qwen3b": {
        "name":      "Qwen2.5-VL 3B",
        "repo_id":   "Qwen/Qwen2.5-VL-3B-Instruct",
        "dtype":     "float16",
        "gpu_mem":   0.40,
        "img_style": "qwen",   # automatic image referencing
    },
    "qwen7b": {
        "name":      "Qwen2.5-VL 7B",
        "repo_id":   "Qwen/Qwen2.5-VL-7B-Instruct",
        "dtype":     "float16",
        "gpu_mem":   0.70,
        "img_style": "qwen",
    },
    "internvl": {
        "name":      "InternVL2.5 4B",
        "repo_id":   "OpenGVLab/InternVL2_5-4B",
        "dtype":     "float16",
        "gpu_mem":   0.80,
        "img_style": "internvl",   # needs "Image-i: <image>" in text
    },
    "smolvlm": {
        "name":      "SmolVLM 2B",
        "repo_id":   "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "dtype":     "float16",
        "gpu_mem":   0.30,
        "img_style": "internvl",   # needs "Image-i: <image>" markers
    },
    "internvl3": {
        "name":      "InternVL3 8B",
        "repo_id":   "OpenGVLab/InternVL3-8B",
        "dtype":     "float16",
        "gpu_mem":   0.85,
        "img_style": "internvl",
    },
    "internvl8b_mpo": {
        "name":      "InternVL2.5 8B MPO",
        "repo_id":   "OpenGVLab/InternVL2_5-8B-MPO",
        "dtype":     "float16",
        "gpu_mem":   0.85,
        "img_style": "internvl",
    },
    "wethink": {
        "name":      "WeThink-Qwen2.5VL 7B",
        "repo_id":   "yangjie-cv/WeThink-Qwen2.5VL-7B",
        "dtype":     "float16",
        "gpu_mem":   0.75,
        "img_style": "qwen",
    },
    "qwen2vl": {
        "name":      "Qwen2-VL 7B",
        "repo_id":   "Qwen/Qwen2-VL-7B-Instruct",
        "dtype":     "float16",
        "gpu_mem":   0.70,
        "img_style": "qwen",
    },
}

VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
VLLM_URL  = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"

# Reasoning modes — each run can evaluate one or several simultaneously
_MODE_CONFIGS = {
    "baseline": {"cot": False, "thinking": False, "max_tokens_override": None},
    "cot":      {"cot": True,  "thinking": False, "max_tokens_override": 300},
    "thinking": {"cot": False, "thinking": True,  "max_tokens_override": 600},
}


# ── Phase detection ───────────────────────────────────────────────────────────

def _detect_phase(sample: dict) -> str:
    """
    Returns the current navigation sub-task for this sample:
      "navigate"  — Empty env, or DoorKey with door already open
      "find_key"  — DoorKey env, key not yet in hand
      "open_door" — DoorKey env, key in hand, door still closed
    """
    if "DoorKey" not in sample.get("env", ""):
        return "navigate"
    if sample.get("agent_carrying") != "key":
        return "find_key"
    if not sample.get("door_open", False):
        return "open_door"
    return "navigate"


# ── Image utilities ───────────────────────────────────────────────────────────

def _encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _image_block(path: Path) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{_encode_image(path)}"},
    }


# ── Prompt building ───────────────────────────────────────────────────────────



def _phase_goal(phase: str, fw: str) -> str:
    """One-paragraph description of the current sub-goal, phase-specific."""
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


# ── Three prompt variants, each phase-aware ───────────────────────────────────

def _v_ifthen(sample: dict, view: str, n_history: int,
              img_style: str, cot: bool, thinking: bool) -> str:
    d     = sample.get("agent_dir_str", "?")
    fw    = _DIR_WORD.get(d, d)
    phase = _detect_phase(sample)
    carry = sample.get("agent_carrying")

    lines = [
        f"Grid navigation. Agent faces {fw}. "
        f"{'Carrying: ' + carry + '.' if carry else 'Carrying: nothing.'}",
        f"Mission: {sample['mission']}",
    ]
    if n_history > 0:
        lines += ["", f"({n_history} history frame(s) shown above — each labeled with step, direction, carrying state and action taken.)"]
    lines += [
        "",
        _phase_goal(phase, fw),
        "",
        "Decision rules — apply the FIRST that matches:",
        f"  → GREEN GOAL directly {fw}                       → action 2 (forward)",
        f"  → KEY directly {fw} and no key in hand           → action 3 (pickup)",
        f"  → DOOR directly {fw} and key in hand             → action 5 (toggle)",
        f"  → Turning right brings you closer to target      → action 1 (turn_right)",
        f"  → Turning left brings you closer to target       → action 0 (turn_left)",
        f"  → Path clear and forward moves toward target     → action 2 (forward)",
        _cot_suffix(cot, thinking),
    ]
    return "\n".join(lines)


def _v_negative(sample: dict, view: str, n_history: int,
                img_style: str, cot: bool, thinking: bool) -> str:
    d     = sample.get("agent_dir_str", "?")
    fw    = _DIR_WORD.get(d, d)
    phase = _detect_phase(sample)
    carry = sample.get("agent_carrying")

    lines = [
        f"Grid agent faces {fw}. "
        f"{'Carrying: ' + carry + '.' if carry else 'Carrying: nothing.'}  "
        f"Mission: {sample['mission']}",
    ]
    if n_history > 0:
        lines += ["", f"({n_history} history frame(s) shown above — each labeled with step, direction, carrying state and action taken.)"]
    lines += [
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


def _v_verbose(sample: dict, view: str, n_history: int,
               img_style: str, cot: bool, thinking: bool) -> str:
    d     = sample.get("agent_dir_str", "?")
    fw    = _DIR_WORD.get(d, d)
    phase = _detect_phase(sample)
    carry = sample.get("agent_carrying")

    view_ctx = (
        f"Full top-down view — every cell is visible. "
        f"The red triangle is the agent, looking {fw}."
        if view == "global" else
        f"7×7 egocentric view centred on the agent. "
        f"The agent is at the bottom-centre, looking {fw}."
    )

    if phase == "find_key":
        phase_section = (
            f"CURRENT TASK — FIND THE KEY:\n"
            f"  You do NOT have the key yet. It is a yellow key somewhere in the grid.\n"
            f"  Navigate toward it. When it is directly {fw} of you, use action 3 (pickup).\n"
            f"  Do not try to open the door until you have the key."
        )
    elif phase == "open_door":
        phase_section = (
            f"CURRENT TASK — OPEN THE DOOR:\n"
            f"  You are carrying the key. Find the coloured door.\n"
            f"  Stand directly in front of it and use action 5 (toggle) to unlock it.\n"
            f"  After it opens, walk through and head to the green goal."
        )
    else:
        phase_section = (
            f"CURRENT TASK — REACH THE GOAL:\n"
            f"  The path to the green square is clear (or the door is already open).\n"
            f"  Navigate directly to the green goal square."
        )

    lines = [
        view_ctx,
        "",
        "GAME MECHANICS:",
        "  Grid objects: walls (dark borders), floor (empty), key (yellow), "
        "door (coloured gate), green goal.",
        "  To win: reach the green goal. In key-door levels: pickup key → toggle door → goal.",
        "  Walls and CLOSED doors block forward movement.",
        "",
        "CURRENT STATE:",
        f"  Mission:   {sample['mission']}",
        f"  Direction: {fw}",
        f"  Carrying:  {carry if carry else 'nothing'}",
    ]
    if n_history > 0:
        lines += ["", f"({n_history} history frame(s) shown above — each labeled with step, direction, carrying state and action taken.)"]
    lines += [
        "",
        phase_section,
        "",
        "AVAILABLE ACTIONS:",
        f"  0 = turn_left   — rotate 90° counter-clockwise (stay in place)",
        f"  1 = turn_right  — rotate 90° clockwise (stay in place)",
        f"  2 = forward     — move one cell {fw} (blocked by walls / closed doors)",
        f"  3 = pickup      — grab the object directly {fw} of you",
        f"  5 = toggle      — open/close the door directly {fw} (requires key)",
        "",
        "WHAT IS THE OPTIMAL ACTION?",
        _cot_suffix(cot, thinking),
    ]
    return "\n".join(lines)


def _v_optimal(sample: dict, view: str, n_history: int,
               img_style: str, cot: bool, thinking: bool) -> str:
    d     = sample.get("agent_dir_str", "?")
    fw    = _DIR_WORD.get(d, d)
    phase = _detect_phase(sample)
    carry = sample.get("agent_carrying")

    view_desc = (
        f"Full top-down view. The red triangle = you, pointing {fw}."
        if view == "global" else
        f"7×7 egocentric view. You are at the bottom-centre, pointing {fw}."
    )

    if phase == "find_key":
        goal_desc  = "the YELLOW KEY (you do not have it yet)"
        phase_goal = (
            f"GOAL: Pick up the YELLOW KEY.\n"
            f"  → When the key is directly {fw} of you and you are facing {fw}: action 3 (pickup).\n"
            f"  → The key is always on YOUR SIDE of the wall, never past the door.\n"
            f"  → If the key is not directly ahead, turn to face it or move toward it.\n"
            f"  → Ignore the door entirely until you have the key."
        )
    elif phase == "open_door":
        goal_desc  = "the DOOR (you carry the key)"
        phase_goal = (
            f"GOAL: Unlock and open the DOOR.\n"
            f"  → When the door is directly {fw} of you and you are facing {fw} AND you carry the key: action 5 (toggle).\n"
            f"  → After opening, walk through toward the green goal."
        )
    else:
        goal_desc  = "the GREEN GOAL square"
        phase_goal = (
            f"GOAL: Reach the GREEN GOAL square.\n"
            f"  → Path is clear. Navigate directly to the green square."
        )

    return f"""{view_desc}

STATE:
  Mission : {sample['mission']}
  Facing  : {fw}
  Carrying: {carry if carry else 'nothing'}

{phase_goal}

ACTIONS:
  0 = turn_left   (rotate 90° CCW — stay in place)
  1 = turn_right  (rotate 90° CW  — stay in place)
  2 = forward     (move one cell {fw} — blocked by walls and closed doors)
  3 = pickup      (grab object directly {fw})
  5 = toggle      (open door directly {fw} — requires key)

REASONING GUIDE:
  Step 1 — Check what is directly {fw}: wall? key? door? goal? empty?
  Step 2 — Locate your target ({goal_desc}). Which direction is it from you?
  Step 3 — Choose your rotation:
      • target is {fw}          → action 2 (forward) if path is clear
      • target is to your LEFT  → turn_right (1) gets you there in 1 step, NOT turn_left
      • target is to your RIGHT → turn_left (0) gets you there in 1 step, NOT turn_right
      • target is BEHIND you    → either turn_left or turn_right (both need 2 steps)
{"  Step 4 — Check the labeled frames above. Are you making progress toward " + goal_desc + "?" + chr(10) + "      If the last few steps show you going back and forth, CHANGE direction." + chr(10) if n_history > 0 else ""}
⚠️  ANTI-BIAS RULES:
  • turn_right (1) is just as valid as turn_left (0). Do NOT systematically prefer one.
  • forward (2) is only correct when you are ALREADY facing your target AND the cell is free.
  • NEVER go forward into a wall or a closed door.
  • pickup (3) ONLY if the KEY is the cell directly {fw}. Not beside you — DIRECTLY {fw}.
  • toggle (5) ONLY if the DOOR is directly {fw} AND you hold the key.
{_cot_suffix(cot, thinking)}"""


PROMPT_VARIANTS = [
    # {"id": 0, "name": "if_then",        "fn": _v_ifthen,   "base_max_out": 16},
    {"id": 1, "name": "negative_rules", "fn": _v_negative, "base_max_out": 16},
    # {"id": 2, "name": "verbose",        "fn": _v_verbose,  "base_max_out": 16},
    {"id": 3, "name": "optimal",        "fn": _v_optimal,  "base_max_out": 600},
]


# ── Response parsing ──────────────────────────────────────────────────────────

def _parse_thinking_response(text: str):
    """Extracts integer from <answer>N</answer>; falls back to standard parsing."""
    m = re.search(r"<answer>\s*(\d+)\s*</answer>", text, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return parse_response(text)


# ── vLLM server management ────────────────────────────────────────────────────

def start_vllm_server(model_cfg: dict, cache_dir: str,
                       max_images: int) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model",                  model_cfg["repo_id"],
        "--dtype",                  model_cfg["dtype"],
        "--gpu-memory-utilization", str(model_cfg["gpu_mem"]),
        "--host",                   VLLM_HOST,
        "--port",                   str(VLLM_PORT),
        "--trust-remote-code",
        "--max-model-len",          "8192",
        "--download-dir",           cache_dir,
    ]
    env = os.environ.copy()
    env["HF_HOME"]               = cache_dir
    env["HUGGINGFACE_HUB_CACHE"] = cache_dir

    print(f"  Starting vLLM server for {model_cfg['name']} "
          f"(max {max_images} images/prompt) ...")
    print(f"  CMD: {' '.join(cmd)}", flush=True)

    return subprocess.Popen(cmd, env=env, stdout=sys.stderr, stderr=sys.stderr)


def wait_for_server(timeout: int = 600) -> bool:
    import urllib.request
    health_url = f"http://{VLLM_HOST}:{VLLM_PORT}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(health_url, timeout=2)
            return True
        except Exception:
            time.sleep(3)
    return False


def stop_vllm_server(proc: subprocess.Popen):
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    time.sleep(5)


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(
    samples: list[dict],
    model_key: str,
    view: str,
    variant: dict,
    dataset_root: Path,
    results_raw_dir: Path,
    max_samples: int | None,
    history_images: int = 0,
    cot: bool = False,
    thinking: bool = False,
    debug: bool = False,
) -> list[dict]:
    """
    Sends all samples to the running vLLM server.

    Each request includes:
      - Up to `history_images` preceding frame(s) (oldest first)
      - The current frame
      - A phase-aware text prompt

    Evaluates against optimal_actions (list) to handle equivalent actions.
    """
    from openai import OpenAI

    client    = OpenAI(base_url=VLLM_URL, api_key="vllm-local")
    model_cfg = MODELS[model_key]
    img_style = model_cfg.get("img_style", "qwen")

    try:
        served_model_name = client.models.list().data[0].id
    except Exception:
        served_model_name = model_cfg["repo_id"]

    subset     = samples[:max_samples] if max_samples else samples
    max_tokens = 600 if thinking else (450 if cot else variant["base_max_out"])
    timeout_s  = 90 if (cot or thinking) else 30

    results = []
    n_ok = n_fail = n_parse = 0

    mode_label = "thinking" if thinking else ("cot" if cot else "baseline")
    pbar = tqdm(
        subset,
        desc=f"  {model_cfg['name']} [{view}] [{variant['name']}] [h{history_images}] [{mode_label}]",
        unit="sample",
    )

    for sample in pbar:
        if not sample.get("oracle_valid", True):
            continue

        # ── Build interleaved content (label → image → label → image → …) ───────
        history     = sample.get("history", [])
        n_hist_use  = min(history_images, len(history))
        recent_hist = history[-n_hist_use:] if n_hist_use > 0 else []
        fallback    = dataset_root / sample[f"{view}_image"]

        content = []
        for i, h in enumerate(recent_hist):
            fw    = _DIR_WORD.get(h.get("agent_dir_str", "?"), h.get("agent_dir_str", "?"))
            carry = "key" if h.get("agent_carrying") == "key" else "nothing"
            act   = h.get("action_name", "?")
            # InternVL needs explicit "Image-i: <image>" in the text
            marker = f"Image-{i+1}: <image>\n" if img_style == "internvl" else ""
            label  = f"{marker}[Step {h['step']:+d} | facing {fw} | carrying {carry} | action taken: {act}]"
            content.append({"type": "text", "text": label})
            img_path = dataset_root / h[f"{view}_image"]
            content.append(_image_block(img_path if img_path.exists() else fallback))

        cur_n    = len(recent_hist) + 1
        cur_mark = f"Image-{cur_n}: <image>\n" if img_style == "internvl" else ""
        content.append({"type": "text", "text": f"{cur_mark}[CURRENT STATE — image {cur_n}]"})
        cur_img_path = dataset_root / sample[f"{view}_image"]
        if cur_img_path.exists():
            content.append(_image_block(cur_img_path))

        actual_n_hist = len(recent_hist)

        # ── Build text prompt — pass actual_n_hist so variants adapt ────────
        user_text = variant["fn"](
            sample, view, actual_n_hist, img_style, cot, thinking
        )
        content.append({"type": "text", "text": user_text})

        messages = [{"role": "user", "content": content}]

        # ── Call the model ────────────────────────────────────────────────────
        t_start = time.time()
        try:
            response = client.chat.completions.create(
                model=served_model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
                timeout=timeout_s,
            )
            raw_output = response.choices[0].message.content or ""
            latency_ms = int((time.time() - t_start) * 1000)
            n_ok += 1
        except Exception:
            raw_output = ""
            latency_ms = -1
            n_fail += 1

        # ── Parse response ────────────────────────────────────────────────────
        predicted_action = (
            _parse_thinking_response(raw_output) if thinking
            else parse_response(raw_output)
        )
        if predicted_action is None:
            n_parse += 1

        optimal_actions = sample.get("optimal_actions", [])
        correct = predicted_action in optimal_actions

        if debug and len(results) < 5:
            icon = "✅" if correct else "❌"
            phase = _detect_phase(sample)
            print(f"  {icon} [{sample.get('id','?')}] phase={phase} "
                  f"hist={actual_n_hist}  raw={repr(raw_output[:60])} "
                  f"→ pred={predicted_action}  oracle={optimal_actions}")

        results.append({
            # Identification
            "sample_id":        sample.get("id", "?"),
            "model":            model_key,
            "model_name":       model_cfg["name"],
            "view":             view,
            "variant":          variant["name"],
            "history_images":   actual_n_hist,
            "mode":             "thinking" if thinking else ("cot" if cot else "baseline"),
            # Sample metadata
            "env":              sample.get("env", ""),
            "complexity":       sample.get("complexity", ""),
            "mission":          sample.get("mission", ""),
            "agent_carrying":   sample.get("agent_carrying"),
            "door_open":        sample.get("door_open", False),
            "history_type":     sample.get("history_type", "N/A"),
            "phase":            _detect_phase(sample),
            # Oracle
            "optimal_actions":  optimal_actions,
            "action_names":     sample.get("action_names", []),
            # Prediction
            "raw_output":       raw_output.strip(),
            "predicted_action": predicted_action,
            "correct":          correct,
            "parse_failed":     predicted_action is None,
            "latency_ms":       latency_ms,
        })

        n_done    = len(results)
        n_correct = sum(r["correct"] for r in results)
        pbar.set_postfix(
            acc=f"{100*n_correct/max(n_done,1):.1f}%",
            fail=n_fail,
            parse_err=n_parse,
        )

    pbar.close()

    # ── Save raw results ──────────────────────────────────────────────────────
    suffix  = f"_h{history_images}" if history_images > 0 else ""
    suffix += "_cot"     if cot     else ""
    suffix += "_think"   if thinking else ""
    out_path = results_raw_dir / f"{model_key}_{view}_{variant['name']}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    n_done    = len(results)
    n_correct = sum(r["correct"] for r in results)
    print(f"  → [{variant['name']}] Acc: {n_correct}/{n_done} "
          f"({100*n_correct/max(n_done,1):.1f}%)  "
          f"API fail: {n_fail}  parse err: {n_parse}  → {out_path.name}")

    return results


# ── Summary ───────────────────────────────────────────────────────────────────

def compute_summary(all_results: list[dict]) -> dict:
    """Accuracy breakdowns per (model, view, variant, hist, mode), complexity, action, phase."""
    summary = {}
    df      = pd.DataFrame(all_results)

    group_cols = ["model", "view", "variant", "history_images", "mode"]
    for keys, grp in df.groupby(group_cols):
        model, view, variant, n_hist, mode = keys
        key = f"{model}_{view}_{variant}_h{n_hist}_{mode}"

        by_complexity = {
            c: {"accuracy": round(s["correct"].mean(), 4), "n": len(s)}
            for c, s in grp.groupby("complexity")
        }
        exploded = grp.explode("action_names")
        by_action = {
            a: {"accuracy": round(s["correct"].mean(), 4), "n": len(s)}
            for a, s in exploded.groupby("action_names")
            if a and str(a) != "nan"
        }
        by_phase = {
            p: {"accuracy": round(s["correct"].mean(), 4), "n": len(s)}
            for p, s in grp.groupby("phase")
        }
        by_history_type = {
            ht: {"accuracy": round(s["correct"].mean(), 4), "n": len(s)}
            for ht, s in grp.groupby("history_type")
        }

        valid_lat = grp[grp["latency_ms"] > 0]["latency_ms"]

        summary[key] = {
            "model":            model,
            "model_name":       MODELS[model]["name"],
            "view":             view,
            "variant":          variant,
            "history_images":   int(n_hist),
            "mode":             mode,
            "accuracy":         round(grp["correct"].mean(), 4),
            "n_samples":        len(grp),
            "n_correct":        int(grp["correct"].sum()),
            "n_parse_failed":   int(grp["parse_failed"].sum()),
            "latency_mean_ms":  int(valid_lat.mean()) if len(valid_lat) else -1,
            "latency_p95_ms":   int(valid_lat.quantile(0.95)) if len(valid_lat) else -1,
            "by_complexity":    by_complexity,
            "by_action":        by_action,
            "by_phase":         by_phase,
            "by_history_type":  by_history_type,
        }

    return summary


def print_summary(summary: dict):
    W    = 88
    rows = sorted(summary.values(),
                  key=lambda x: (x["model"], x["view"], x["variant"],
                                 x["history_images"], x["mode"]))

    print(f"\n{'═'*W}")
    print("  BENCHMARK RESULTS")
    print(f"{'═'*W}")
    print(f"  {'Model':<25} {'View':<8} {'Variant':<16} {'H':>2}  {'Mode':<10}  "
          f"{'Accuracy':>9}  {'N':>5}  {'Lat(ms)':>8}")
    print(f"  {'─'*25} {'─'*8} {'─'*16} {'─'*2}  {'─'*10}  {'─'*9}  {'─'*5}  {'─'*8}")
    for r in rows:
        print(f"  {r['model_name']:<25} {r['view']:<8} {r['variant']:<16} "
              f"{r['history_images']:>2}  {r['mode']:<10}  "
              f"{100*r['accuracy']:>8.1f}%  {r['n_samples']:>5}  "
              f"{r['latency_mean_ms']:>7}ms")

    # Per-phase breakdown
    print(f"\n{'═'*W}")
    print("  PER-PHASE ACCURACY")
    print(f"{'═'*W}")
    print(f"  {'Model / Variant / Mode':<50} {'navigate':>10} {'find_key':>10} {'open_door':>10}")
    print(f"  {'─'*50} {'─'*10} {'─'*10} {'─'*10}")
    for r in rows:
        label = f"{r['model_name']} / {r['variant']} h={r['history_images']} [{r['mode']}]"
        phases = r.get("by_phase", {})
        def _pct(p):
            info = phases.get(p)
            return f"{100*info['accuracy']:>9.1f}%" if info else f"{'N/A':>10}"
        print(f"  {label:<50} {_pct('navigate')} {_pct('find_key')} {_pct('open_door')}")

    # Per-action breakdown (global view)
    global_rows = [r for r in rows if r["view"] == "global"]
    all_actions = sorted({a for r in global_rows for a in r["by_action"]})
    if all_actions:
        print(f"\n{'═'*W}")
        print("  PER-ACTION ACCURACY  (global view)")
        print(f"{'═'*W}")
        lw = 44
        print(f"  {'Model / Variant':<{lw}}" + "".join(f"{a:>12}" for a in all_actions))
        print(f"  {'─'*lw}" + "".join("─"*12 for _ in all_actions))
        for r in global_rows:
            label = f"{r['model_name']} / {r['variant']} h={r['history_images']} [{r['mode']}]"
            row   = f"  {label:<{lw}}"
            for a in all_actions:
                acc = r["by_action"].get(a, {}).get("accuracy")
                row += f"{100*acc:>11.1f}%" if acc is not None else f"{'N/A':>12}"
            print(row)

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def run_benchmark(
    dataset_path: str,
    results_dir: str,
    cache_dir: str,
    model_keys: list[str],
    views: list[str],
    max_samples: int | None,
    history_images: list[int] = None,
    modes: list[str] = None,
    debug: bool = False,
):
    if modes is None:
        modes = ["baseline"]
    if history_images is None:
        history_images = [0]

    dataset_root    = Path(dataset_path).parent
    results_path    = Path(results_dir)
    results_raw_dir = results_path / "raw"
    results_raw_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"]               = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir

    with open(dataset_path) as f:
        samples = json.load(f)

    max_images_per_prompt = max(history_images) + 1

    print(f"\n  Dataset loaded   : {len(samples)} samples from {dataset_path}")
    print(f"  History images   : {history_images}  (max {max_images_per_prompt} images/prompt)")
    print(f"  Modes            : {modes}")
    print(f"  Prompt variants  : {[v['name'] for v in PROMPT_VARIANTS]}")

    from collections import Counter
    phases = [_detect_phase(s) for s in samples]
    for phase, cnt in sorted(Counter(phases).items()):
        print(f"    {phase:<12} : {cnt} samples")

    all_results = []

    for model_key in model_keys:
        model_cfg = MODELS[model_key]
        print(f"\n{'═'*65}")
        print(f"  MODEL: {model_cfg['name']}  ({model_cfg['repo_id']})")
        print(f"{'═'*65}")

        proc = start_vllm_server(model_cfg, cache_dir, max_images_per_prompt)
        print("  Waiting for server to be ready (up to 10 min) ...")
        if not wait_for_server(timeout=600):
            print("  Server did not start in time. Skipping this model.")
            stop_vllm_server(proc)
            continue
        print("  ✅  Server ready.")

        for view in views:
            for variant in PROMPT_VARIANTS:
                for mode_name in modes:
                    for n_hist in history_images:
                        mode_cfg = _MODE_CONFIGS[mode_name]
                        results = run_inference(
                            samples=samples,
                            model_key=model_key,
                            view=view,
                            variant=variant,
                            dataset_root=dataset_root,
                            results_raw_dir=results_raw_dir,
                            max_samples=max_samples,
                            history_images=n_hist,
                            cot=mode_cfg["cot"],
                            thinking=mode_cfg["thinking"],
                            debug=debug,
                        )
                        all_results.extend(results)

        print("  Stopping vLLM server ...")
        stop_vllm_server(proc)
        print("  GPU memory released.")

    if not all_results:
        print("\n  No results collected — exiting.")
        return

    df       = pd.DataFrame(all_results)
    csv_path = results_path / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV saved → {csv_path}")

    summary      = compute_summary(all_results)
    summary_path = results_path / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved → {summary_path}")

    print_summary(summary)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Run VLM benchmark on MiniGrid dataset using vLLM"
    )
    p.add_argument("--dataset",        type=str, default="./dataset/dataset.json")
    p.add_argument("--results",        type=str, default="./results")
    p.add_argument("--cache_dir",      type=str,
                   default=os.environ.get("HF_HOME", "./hf_cache"))
    p.add_argument("--models",         nargs="+", default=list(MODELS.keys()),
                   choices=list(MODELS.keys()))
    p.add_argument("--views",          nargs="+", default=["global", "partial"],
                   choices=["global", "partial"])
    p.add_argument("--max_samples",    type=int,  default=None)
    p.add_argument("--history_images", nargs="+", type=int, default=[0],
                   help="Number(s) of history frames to compare. Pass several values to "
                        "evaluate them in one run: --history_images 0 3 5  "
                        "vLLM is launched with max(values)+1 images/prompt.")
    p.add_argument("--modes", nargs="+",
                   default=["baseline"],
                   choices=list(_MODE_CONFIGS.keys()),
                   help="Reasoning mode(s) to evaluate. Pass several to compare them "
                        "in one run: --modes baseline cot thinking  "
                        "(baseline=16 tok, cot=300 tok, thinking=600 tok + <answer> parsing).")
    p.add_argument("--debug",   action="store_true",
                   help="Print the first 5 predictions per (model, view, variant, mode) run.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        dataset_path=args.dataset,
        results_dir=args.results,
        cache_dir=args.cache_dir,
        model_keys=args.models,
        views=args.views,
        max_samples=args.max_samples,
        history_images=args.history_images,  # now a list
        modes=args.modes,
        debug=args.debug,
    )
