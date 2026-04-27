import pandas as pd
from pathlib import Path
import numpy as np
from ..run_benchmark import PROMPT_VARIANTS

def _debug_probe(client, served_model_name: str, samples: list[dict],
                 view: str, dataset_root: Path, model_key: str):
    """
    Runs before the main inference loop when --debug is set.
    `samples` is a list of up to 5 valid samples.

    Sections:
      1. Prints the full text prompt for samples[0] (no base64).
      2. Open-ended "what do you see?" on samples[0].
      3. With vs without image comparison on samples[0].
      4. All prompt variants tested on all samples, with per-sample
         results and a final ranked accuracy table.
    """
    from utils.prompt_builder import build_prompt

    sep  = "─" * 70
    sep2 = "═" * 70

    sample = samples[0]  # sections 1-3 always use the first sample

    print(f"\n{sep2}")
    print("  DEBUG PROBE")
    print(sep2)

    messages, meta = build_prompt(sample, view=view, model_key=model_key,
                                  dataset_root=dataset_root)

    # ── 1. Print text prompt ──────────────────────────────────────────────────
    print("\n  [1] Text prompt sent to model (first sample):")
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            print(f"    role={msg['role']}: {content}")
        else:
            for part in content:
                if part["type"] == "text":
                    print(f"    role={msg['role']} [text]:\n{part['text']}")
                elif part["type"] == "image_url":
                    b64 = part["image_url"]["url"].split(",", 1)[-1]
                    print(f"    role={msg['role']} [image]: {len(b64)} base64 chars "
                          f"(~{len(b64)*3//4//1024} KB)")

    def _ask(msgs, max_tokens=64):
        try:
            r = client.chat.completions.create(
                model=served_model_name, messages=msgs,
                max_tokens=max_tokens, temperature=0.0, timeout=30,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            return f"ERROR: {e}"

    def _extract_img_block(msgs):
        for msg in msgs:
            if isinstance(msg["content"], list):
                for part in msg["content"]:
                    if part["type"] == "image_url":
                        return part
        return None

    img_block = _extract_img_block(messages)

    # ── 2. Open-ended "what do you see?" ─────────────────────────────────────
    print("\n  [2] What does the model see in the image? (first sample)")
    if img_block:
        see_msgs = [{"role": "user", "content": [
            img_block,
            {"type": "text", "text": (
                "Describe what you see in this image in 1-2 sentences, "
                "and tell me in which direction the green square is relative to the red triangle. "
                "Also give me the one action (to reach the green square quickly) "
                "you would take based on what you see, without any explanation."
            )},
        ]}]
        out_see = _ask(see_msgs, max_tokens=200)
        print(f"    → {out_see}")
        if not out_see or "ERROR" in out_see:
            print("Empty/error response — vision encoder may not be loaded.")
    else:
        print("    (no image block found in prompt)")

    # ── 3. With vs without image on the actual task ───────────────────────────
    print("\n  [3] Task prompt WITH image vs WITHOUT image (first sample):")

    out_with = _ask(messages, max_tokens=16)

    text_only_messages = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            text_only_messages.append(msg)
        else:
            text_parts = [p for p in content if p["type"] == "text"]
            text_only_messages.append({
                "role": msg["role"],
                "content": text_parts[0]["text"] if len(text_parts) == 1 else text_parts,
            })
    out_without = _ask(text_only_messages, max_tokens=16)

    print(f"    WITH image    → {repr(out_with)}")
    print(f"    WITHOUT image → {repr(out_without)}")

    oracle0 = meta["optimal_actions"]
    if out_with == out_without:
        print("    ⚠️  IDENTICAL — model is ignoring the image.")
    else:
        print("    ✅  Outputs differ — image IS influencing the model.")
    print(f"    Oracle for this sample: {oracle0} ({meta.get('action_names', [])})")

    # ── 4. Prompt variant comparison — all variants × all samples ────────────
    if img_block is None:
        print(sep2)
        return

    n_samples = len(samples)
    print(f"\n{sep}")
    print(f"  [4] Prompt variant comparison — {n_samples} sample(s), {view} view")
    print(sep)

    def _parse(raw):
        for token in (raw or "").split():
            t = token.strip(".,;:()")
            if t.isdigit() and int(t) in range(6):
                return int(t)
        if raw:
            kw = {"left": 0, "right": 1, "forward": 2, "pickup": 3, "toggle": 5, "open": 5}
            for word, act in kw.items():
                if word in raw.lower():
                    return act
        return None

    scores = {v["id"]: {"correct": 0, "total": 0} for v in PROMPT_VARIANTS}

    hdr     = f"  {'ID':<3} {'Name':<16} {'~tok':>5}  {'Raw output':<30}  {'Pred':>4}  Result"
    hdr_sep = f"  {'──':<3} {'────────────────':16} {'─────':>5}  {'─'*30}  {'────':>4}  ──────"

    for s in samples:
        s_msgs, s_meta = build_prompt(s, view=view, model_key=model_key,
                                      dataset_root=dataset_root)
        s_img  = _extract_img_block(s_msgs)
        oracle = s_meta["optimal_actions"]

        print(f"\n  ── Sample {s_meta['sample_id']}  oracle={oracle}  "
              f"({s_meta.get('action_names', [])})  mission: {s_meta['mission']}")
        print(hdr)
        print(hdr_sep)

        for v in PROMPT_VARIANTS:
            user_text = v["fn"](s, view)
            if s_img:
                v_msgs = [{"role": "user", "content": [
                    s_img,
                    {"type": "text", "text": user_text},
                ]}]
            else:
                v_msgs = [{"role": "user", "content": user_text}]

            raw       = _ask(v_msgs, max_tokens=v["max_out"])
            predicted = _parse(raw)
            correct   = predicted in oracle

            scores[v["id"]]["total"] += 1
            if correct:
                scores[v["id"]]["correct"] += 1

            icon     = "✅" if correct else ("❌" if predicted is not None else "⚠️ ")
            raw_disp = repr(raw[:28]) if raw else "''"
            print(f"  {v['id']:<3} {v['name']:<16} {v['approx_tokens']:>5}  "
                  f"{raw_disp:<30}  {str(predicted):>4}  {icon}")

    # ── Final ranked summary ──────────────────────────────────────────────────
    print(f"\n{sep2}")
    print(f"  VARIANT STATS — {n_samples} sample(s), {view} view, model={model_key}")
    print(sep2)
    print(f"  {'Rank':<5} {'ID':<4} {'Name':<16} {'~tok':>5}  {'Correct':>9}  {'Accuracy':>9}")
    print(f"  {'────':<5} {'──':<4} {'────────────────':16} {'─────':>5}  {'───────':>9}  {'────────':>9}")

    ranked = sorted(
        PROMPT_VARIANTS,
        key=lambda v: (
            -scores[v["id"]]["correct"] / max(scores[v["id"]]["total"], 1),
            v["approx_tokens"],
        ),
    )
    for rank, v in enumerate(ranked, 1):
        sc   = scores[v["id"]]
        acc  = sc["correct"] / max(sc["total"], 1)
        frac = f"{sc['correct']}/{sc['total']}"
        print(f"  {rank:<5} {v['id']:<4} {v['name']:<16} {v['approx_tokens']:>5}  "
              f"{frac:>9}  {100*acc:>8.1f}%")

    print(sep2)