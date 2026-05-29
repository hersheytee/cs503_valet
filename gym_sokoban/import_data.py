import wandb
import pandas as pd

ENTITY = "hersheytee"
PROJECT = "cs503-sokoban"
GROUPS = [
    "sokoban_randomized_oracle_500k",
    "sokoban_randomized_oracle_extra_500k",
    "sokoban_linear_cost_500k",
    "worldcoder_2m",
]

api = wandb.Api()
all_rows = []

for group in GROUPS:
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": group})
    for run in runs:
        cfg = run.config
        summary = run.summary._json_dict

        all_rows.append({
            "run_name": run.name,
            "group": group,
            "state": run.state,
            "env_id": cfg.get("env_id"),
            "seed": cfg.get("seed"),
            "oracle_cost": cfg.get("oracle_cost"),
            "oracle_cost_final": cfg.get("oracle_cost_final"),
            "oracle_accuracy": cfg.get("oracle_accuracy"),
            "no_oracle": cfg.get("no_oracle"),
            "total_timesteps": cfg.get("total_timesteps"),
            "final_success_rate": summary.get("episode/success_rate"),
            "final_return": summary.get("episode/return"),
            "final_guided_pct": summary.get("episode/guided_pct"),
            "final_queries_per_ep": summary.get("episode/queries_per_ep"),
            "final_oracle_correct_rate": summary.get("episode/oracle_correct_rate"),
        })

df = pd.DataFrame(all_rows)
df.to_csv("wandb_sokoban_summary.csv", index=False)
print(df.sort_values(["group", "oracle_accuracy", "oracle_cost"]))