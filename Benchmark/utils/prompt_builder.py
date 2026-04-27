import base64
from pathlib import Path
from typing import Literal

#  Action definitions (must match dataset) 
ACTIONS = {
    0: ("turn_left",  "Turn left  — rotate 90° counter-clockwise, stay in place"),
    1: ("turn_right", "Turn right — rotate 90° clockwise, stay in place"),
    2: ("forward",    "Move forward — move one cell in the direction you are facing"),
    3: ("pickup",     "Pick up — grab the object directly in front of you"),
    4: ("drop",       "Drop — place the object you are carrying in front of you"),
    5: ("toggle",     "Toggle — open/close the door directly in front of you"),
}

DIR_DESCRIPTION = {
    "→": "RIGHT →",
    "↓": "DOWN ↓",
    "←": "LEFT ←",
    "↑": "UP ↑",
}

# Used to make the "forward" action description concrete
DIR_WORD = {
    "→": "RIGHT",
    "↓": "DOWN",
    "←": "LEFT",
    "↑": "UP",
}

# Models that do NOT support a separate system message 
# Their chat template raises TypeError when content is a list + system message.
# For these, we merge system prompt into the user turn.
SIMPLE_FORMAT_MODELS = {"internvl", "llava", "smolvlm"}

# System prompt 
SYSTEM_PROMPT = """\
You are an expert navigation assistant for a grid-world agent.
Your role is to analyze the agent's current visual observation and recommend \
the single best action to make progress toward the mission goal.

You must respond with ONLY a single integer corresponding to the action number. \
No explanation, no text, no punctuation — just the integer.\
"""

# View-specific context
VIEW_CONTEXT = {
    "global": (
        "The image shows a TOP-DOWN view of the ENTIRE grid. "
        "You can see all objects, walls, doors, keys, and the goal. "
        "The red triangle is the agent — its pointy tip shows which direction it faces. "
        "The agent's exact facing direction is stated in text below: use the text as ground truth."
    ),
    "partial": (
        "The image shows the agent's EGOCENTRIC view: a 7×7 tile window "
        "of what is directly in front of and around the agent. "
        "The agent is always at the bottom-center of the image, facing UPWARD in the frame — "
        "so the top of the image is what is directly ahead of the agent. "
        "Cells outside the field of view appear dark grey."
    ),
}


# Image Loading

def load_image_b64(image_path: str | Path) -> str:
    """Loads an image from disk and returns it as a base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# State description

def _describe_state(sample: dict, view: str) -> str:
    """Builds a natural language description of the agent's current state."""
    dir_str  = sample.get("agent_dir_str", "?")
    carrying = sample.get("agent_carrying")
    mission  = sample.get("mission", "")

    lines = []
    lines.append(f"Mission: {mission}")
    lines.append(
        f"Agent is facing: {DIR_DESCRIPTION.get(dir_str, dir_str)} "
        f"— moving forward will move the agent {DIR_WORD.get(dir_str, '')} in the grid."
    )

    if carrying:
        lines.append(f"Agent is currently carrying: a {carrying}")
    else:
        lines.append("Agent is not carrying anything")

    if view == "partial":
        lines.append(
            "Note: you only see a partial 7×7 view around the agent. "
            "Objects outside this window are not visible."
        )

    return "\n".join(lines)


# Action menu

def _build_action_menu(carrying: str | None, dir_str: str = "?") -> str:
    """
    Returns a formatted list of available actions.
    Hides 'drop' if the agent is not carrying anything.
    Action 2 (forward) is annotated with the concrete direction.
    """
    forward_dir = DIR_WORD.get(dir_str, "")
    lines = ["Available actions:"]
    for idx, (name, description) in ACTIONS.items():
        if idx == 4 and carrying is None:
            continue
        if idx == 2 and forward_dir:
            lines.append(f"  {idx}: Move forward — move one cell {forward_dir} (the direction the agent is facing)")
        else:
            lines.append(f"  {idx}: {description}")
    lines.append("\nRespond with the action number only !")
    return "\n".join(lines)

# Main prompt builder

def build_prompt(
    sample: dict,
    view: Literal["global", "partial"],
    model_key: str = "qwen3b",
    dataset_root: str | Path = ".",
) -> tuple[list[dict], dict]:
    """
    Builds a chat-format prompt for a single dataset sample.

    Parameters
    ----------
    sample : dict
        One entry from dataset.json.
    view : "global" or "partial"
        Which image to use.
    model_key : str
        Key from MODELS dict — used to pick the right prompt format.
        Models in SIMPLE_FORMAT_MODELS get a merged user-only message.
    dataset_root : path
        Root directory of the dataset (images are relative to this).

    Returns
    -------
    messages : list[dict]
        Chat messages in OpenAI format, ready for vLLM.
    meta : dict
        Prompt metadata for logging and evaluation.
    """
    dataset_root = Path(dataset_root)
    carrying     = sample.get("agent_carrying")

    # Load image
    img_key  = "global_image" if view == "global" else "partial_image"
    img_path = dataset_root / sample[img_key]
    img_b64  = load_image_b64(img_path)

    image_block = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
    }

    # Build text parts
    dir_str   = sample.get("agent_dir_str", "?")
    user_text = (
        f"{VIEW_CONTEXT[view]}\n\n"
        f"{_describe_state(sample, view)}\n\n"
        f"{_build_action_menu(carrying, dir_str)}"
    )

    #  Per-model message format 
    if model_key in SIMPLE_FORMAT_MODELS:
        # InternVL / LLaVA / SmolVLM: no system message, everything in user turn
        # System prompt merged into user text to avoid chat template errors
        messages = [
            {
                "role": "user",
                "content": [
                    image_block,
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT + "\n\n" + user_text,
                    },
                ],
            }
        ]
    else:
        # Gemma / Qwen: standard OpenAI format with separate system message
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    image_block,
                    {
                        "type": "text",
                        "text": user_text,
                    },
                ],
            },
        ]

    #  optimal_actions: list of all equally valid actions
    optimal_actions = sample.get("optimal_actions")
    if optimal_actions is None:
        # Fallback for old samples that only have optimal_action (int)
        fallback = sample.get("optimal_action")
        optimal_actions = [fallback] if fallback is not None else []

    meta = {
        "sample_id":       sample["id"],
        "view":            view,
        "model_key":       model_key,
        "env":             sample["env"],
        "complexity":      sample.get("complexity", "unknown"),
        "mission":         sample.get("mission", ""),
        "optimal_actions": optimal_actions,   # list of equally valid actions
        "action_names":    sample.get("action_names", []),
        "agent_carrying":  carrying,
        "oracle_valid":    sample.get("oracle_valid", True),
    }

    return messages, meta


# Response parser in case the model does not follow instructions perfectly

def parse_response(raw_output: str) -> int | None:
    """
    Extracts the action integer from a VLM response.

    Handles:
      - Clean integer:       "2"                   → 2
      - With whitespace:     "  3 \n"               → 3
      - With explanation:    "2 (move forward)"     → 2
      - Spelled out:         "forward"              → 2  (fuzzy fallback)
      - Natural sentence:    "I recommend forward"  → 2  (fuzzy fallback)

    Returns None if no valid action can be extracted.
    """
    if not raw_output:
        return None

    text = raw_output.strip()

    # 1. Try to find a leading integer
    for token in text.split():
        token = token.strip(".,;:()")
        if token.isdigit():
            action = int(token)
            if action in ACTIONS:
                return action

    # 2. Fuzzy fallback — match action name keywords
    text_lower = text.lower()
    keyword_map = {
        "turn left":  0,
        "turn_left":  0,
        "left":       0,
        "turn right": 1,
        "turn_right": 1,
        "right":      1,
        "forward":    2,
        "move":       2,
        "ahead":      2,
        "straight":   2,
        "pickup":     3,
        "pick up":    3,
        "pick":       3,
        "grab":       3,
        "collect":    3,
        "drop":       4,
        "place":      4,
        "toggle":     5,
        "open":       5,
        "unlock":     5,
        "door":       5,
    }
    # Check longer phrases first to avoid partial matches
    for keyword, action_id in sorted(keyword_map.items(), key=lambda x: -len(x[0])):
        if keyword in text_lower:
            return action_id

    return None


# Self test

if __name__ == "__main__":
    fake_sample = {
        "id":              "00001",
        "env":             "MiniGrid-DoorKey-8x8-v0",
        "complexity":      "medium",
        "mission":         "use the key to open the door and then get to the goal",
        "agent_pos":       [3, 4],
        "agent_dir":       1,
        "agent_dir_str":   "↓",
        "global_image":    "images/00001_global.png",
        "partial_image":   "images/00001_partial.png",
        "optimal_actions": [3],
        "action_name":     "pickup",
        "agent_carrying":  None,
        "oracle_valid":    True,
    }

    print("=" * 60)
    print("FORMAT TEST — standard (Gemma/Qwen)")
    print("=" * 60)
    msgs, meta = build_prompt(fake_sample, "global", model_key="qwen3b")
    print(f"  Nb messages     : {len(msgs)}")
    print(f"  Roles           : {[m['role'] for m in msgs]}")
    print(f"  User content    : {[c['type'] for c in msgs[-1]['content']]}")
    print(f"  optimal_actions : {meta['optimal_actions']}")

    print("\n" + "=" * 60)
    print("FORMAT TEST — simple (InternVL/LLaVA/SmolVLM)")
    print("=" * 60)
    msgs2, _ = build_prompt(fake_sample, "global", model_key="internvl")
    print(f"  Nb messages     : {len(msgs2)}")
    print(f"  Roles           : {[m['role'] for m in msgs2]}")
    print(f"  User content    : {[c['type'] for c in msgs2[-1]['content']]}")
    print(f"  System merged   : {'SYSTEM' in msgs2[0]['content'][1]['text'][:20]}")

    print("\n" + "=" * 60)
    print("RESPONSE PARSER TEST")
    print("=" * 60)
    test_cases = [
        ("2",                    2),
        ("  3 \n",               3),
        ("2 (move forward)",     2),
        ("forward",              2),
        ("I recommend forward",  2),
        ("turn left",            0),
        ("turn right",           1),
        ("open the door",        5),
        ("grab the key",         3),
        ("pick up the key",      3),
        ("xyz",                  None),
        ("",                     None),
    ]
    all_ok = True
    for raw, expected in test_cases:
        result = parse_response(raw)
        status = "✅" if result == expected else "❌"
        if result != expected:
            all_ok = False
        print(f"  {status}  parse_response({repr(raw):30s}) → {result}  (expected {expected})")
    print(f"\n{'All tests passed ✅' if all_ok else 'Some tests failed ❌'}")