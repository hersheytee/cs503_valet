"""
Variants de prompts pour PromptBenchmark.

Tous les variants partent du MÊME socle "scaffolded" (inspiré de l'ancien
_v_verbose qui marchait), et n'ajoutent qu'UNE chose :
  - history       : l'historique des 5 dernières actions
  - cot           : une demande de raisonnement step-by-step
  - history_cot   : les deux

Ça permet d'isoler PROPREMENT l'effet de chaque ajout.

Chaque fonction retourne UNIQUEMENT le texte du prompt utilisateur.
L'image est gérée par run_benchmark.py via build_prompt().
"""

import sys
sys.path.append("../Benchmark")
from utils.prompt_builder import DIR_WORD


# ── Helpers ──────────────────────────────────────────────────────────────────

def _format_history(history):
    """Format propre : 'turn_left → forward → turn_right → forward → forward'."""
    if not history:
        return "(no prior actions)"
    return " → ".join(history)


def _scaffolded_base(s, view="global"):
    d        = s.get("agent_dir_str", "?")
    fw       = DIR_WORD.get(d, d)
    carrying = s.get("agent_carrying")
    carry_line = (
        f"The agent is currently carrying: a {carrying}."
        if carrying else "The agent is not carrying anything."
    )
    view_ctx = (
        "The image is a full top-down view of the grid. "
        f"Every cell is visible. The red triangle is the agent, looking {fw}."
    ) if view == "global" else (
        f"The image shows a 7×7 egocentric view. "
        f"The agent is at the bottom-center, looking {fw}."
    )
    return (
        f"{view_ctx}\n\n"
        f"GAME MECHANICS:\n"
        f"- The grid contains: walls (dark border cells), floor (empty cells), "
        f"keys (yellow objects), doors (colored gates), and a green goal square.\n"
        f"- To win: reach the green goal square.\n"
        f"- If there is a door: you must first PICK UP the key (action 3 when facing it), "
        f"then OPEN the door (action 5 when facing it with key in hand), then go through.\n"
        f"- Walls and closed doors block movement.\n\n"
        f"CURRENT STATE:\n"
        f"- Mission: {s['mission']}\n"
        f"- Agent facing: {fw}\n"
        f"- {carry_line}\n\n"
        f"ACTIONS:\n"
        f"  0 = turn_left   — rotate 90° counter-clockwise (no movement)\n"
        f"  1 = turn_right  — rotate 90° clockwise (no movement)\n"
        f"  2 = forward     — move one cell {fw} (blocked by walls/closed doors)\n"
        f"  3 = pickup      — grab object in the cell directly {fw} of you\n"
        f"  5 = toggle      — open/close door in the cell directly {fw} of you\n"
    )


# ── Variants ─────────────────────────────────────────────────────────────────

# 1. Baseline scaffolded — référence (= ancien verbose, sans rien d'ajouté)
def v_baseline(s, view="global"):
    return (
        _scaffolded_base(s, view)
        + "\nWHAT IS THE OPTIMAL BEST ACTION? REPLY WITH ONE INTEGER ONLY."
    )


# 2. Scaffolded + historique brut
def v_history(s, view="global"):
    history = _format_history(s.get("action_history", []))
    return (
        _scaffolded_base(s, view)
        + f"\nYOUR LAST ACTIONS (most recent last): {history}\n"
        + "\nWHAT IS THE OPTIMAL NEXT ACTION? REPLY WITH ONE INTEGER ONLY."
    )


# 3. Scaffolded + historique + invitation à raisonner sur lui
def v_history_reason(s, view="global"):
    history = _format_history(s.get("action_history", []))
    return (
        _scaffolded_base(s, view)
        + f"\nYOUR RECENT ACTIONS led you to the current state: {history}\n"
        + "Consider: did these actions make progress? Should you continue or change strategy?\n"
        + "\nWHAT IS THE OPTIMAL NEXT ACTION? REPLY WITH ONE INTEGER ONLY."
    )


# 4. Scaffolded + Chain of Thought (sans historique)
def v_cot(s, view="global"):
    return (
        _scaffolded_base(s, view)
        + "\nThink step by step:\n"
        + "1. Where is the green goal relative to the agent?\n"
        + "2. What is blocking the path (wall, door, key)?\n"
        + "3. What single action makes the most progress toward the mission?\n"
        + "\nAfter reasoning, REPLY WITH ONE INTEGER ONLY on the last line."
    )


# 5. Scaffolded + historique + CoT (cumul)
def v_history_cot(s, view="global"):
    history = _format_history(s.get("action_history", []))
    return (
        _scaffolded_base(s, view)
        + f"\nYOUR RECENT ACTIONS: {history}\n"
        + "\nThink step by step:\n"
        + "1. Did your past actions make progress toward the goal?\n"
        + "2. Where is the green goal relative to your current position?\n"
        + "3. What is blocking the path now?\n"
        + "4. What single action makes the most progress?\n"
        + "\nAfter reasoning, REPLY WITH ONE INTEGER ONLY on the last line."
    )


# ── Registry ─────────────────────────────────────────────────────────────────

PROMPT_VARIANTS = [
    {"id": "baseline",       "name": "baseline",       "fn": v_baseline,       "max_out": 8,   "approx_tokens": 310},
    {"id": "history",        "name": "history",        "fn": v_history,        "max_out": 8,   "approx_tokens": 350},
    {"id": "history_reason", "name": "history_reason", "fn": v_history_reason, "max_out": 8,   "approx_tokens": 380},
    {"id": "cot",            "name": "cot",            "fn": v_cot,            "max_out": 128, "approx_tokens": 360},
    {"id": "history_cot",    "name": "history_cot",    "fn": v_history_cot,    "max_out": 128, "approx_tokens": 410},
]


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fake = {
        "mission":        "use the key to open the door and then get to the goal",
        "agent_dir_str":  "→",
        "agent_carrying": None,
        "action_history": ["forward", "turn_right", "forward", "forward", "turn_left"],
    }
    for v in PROMPT_VARIANTS:
        print("=" * 78)
        print(f"VARIANT: {v['name']}  (max_out={v['max_out']})")
        print("=" * 78)
        print(v["fn"](fake))
        print()