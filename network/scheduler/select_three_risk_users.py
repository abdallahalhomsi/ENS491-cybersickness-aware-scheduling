import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_PATH = SCRIPT_DIR / "risk_scores_all_sessions.json"
OUTPUT_PATH = SCRIPT_DIR / "risk_scores.json"


with open(INPUT_PATH, "r") as f:
    scores = json.load(f)


items = list(scores.items())

low_user, low_score = min(
    items,
    key=lambda x: x[1]
)

medium_user, medium_score = min(
    items,
    key=lambda x: abs(x[1] - 0.50)
)

high_user, high_score = max(
    items,
    key=lambda x: x[1]
)


selected = {
    low_user: low_score,
    medium_user: medium_score,
    high_user: high_score
}


with open(OUTPUT_PATH, "w") as f:
    json.dump(
        selected,
        f,
        indent=2
    )


print("Selected users:")
print(f"LOW    : {low_user} -> {low_score:.3f}")
print(f"MEDIUM : {medium_user} -> {medium_score:.3f}")
print(f"HIGH   : {high_user} -> {high_score:.3f}")

print()
print("Saved to:")
print(OUTPUT_PATH)