import pandas as pd
from pathlib import Path


RAW_DIR = Path("data/raw/Questset")
PROCESSED_DIR = Path("data/processed")


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

master = pd.read_csv(PROCESSED_DIR / "questset_master_sessions.csv")
ssq = pd.read_csv(RAW_DIR / "SSQ.csv")


# ---------------------------------------------------------
# Encode SSQ severity
# None/blank = 0
# Slight = 1
# Moderate = 2
# Severe = 3
# ---------------------------------------------------------

severity_map = {
    "Slight": 1,
    "Moderate": 2,
    "Severe": 3
}


symptom_cols = [
    "General discomfort",
    "Fatigue",
    "Headache",
    "Eyestrain",
    "Difficulty focusing",
    "Increased salivation",
    "Nausea",
    "Difficulty concentrating",
    "Fullness of the head",
    "Blurred vision",
    "Dizziness (eyes closed)",
    "Dizziness (eyes open)",
    "Vertigo",
    "Stomach awareness",
    "Burping",
    "Sweating"
]


for col in symptom_cols:
    ssq[col] = ssq[col].map(severity_map).fillna(0).astype(int)


# ---------------------------------------------------------
# Standard Kennedy SSQ scoring
# ---------------------------------------------------------

nausea_items = [
    "General discomfort",
    "Increased salivation",
    "Sweating",
    "Nausea",
    "Difficulty concentrating",
    "Stomach awareness",
    "Burping"
]

oculomotor_items = [
    "General discomfort",
    "Fatigue",
    "Headache",
    "Eyestrain",
    "Difficulty focusing",
    "Difficulty concentrating",
    "Blurred vision"
]

disorientation_items = [
    "Difficulty focusing",
    "Nausea",
    "Fullness of the head",
    "Blurred vision",
    "Dizziness (eyes closed)",
    "Dizziness (eyes open)",
    "Vertigo"
]


ssq["ssq_nausea"] = ssq[nausea_items].sum(axis=1) * 9.54
ssq["ssq_oculomotor"] = ssq[oculomotor_items].sum(axis=1) * 7.58
ssq["ssq_disorientation"] = ssq[disorientation_items].sum(axis=1) * 13.92

ssq["ssq_total"] = ssq[symptom_cols].sum(axis=1) * 3.74


# ---------------------------------------------------------
# Prepare lookup table
# ---------------------------------------------------------

ssq_lookup = ssq[
    [
        "ID",
        "Questionnaire number",
        "ssq_nausea",
        "ssq_oculomotor",
        "ssq_disorientation",
        "ssq_total"
    ]
].copy()


# ---------------------------------------------------------
# Merge PRE questionnaire
# ---------------------------------------------------------

pre = ssq_lookup.rename(columns={
    "ID": "participant_id",
    "Questionnaire number": "pre_questionnaire",
    "ssq_nausea": "pre_ssq_nausea",
    "ssq_oculomotor": "pre_ssq_oculomotor",
    "ssq_disorientation": "pre_ssq_disorientation",
    "ssq_total": "pre_ssq_total"
})

master = master.merge(
    pre,
    on=["participant_id", "pre_questionnaire"],
    how="left"
)


# ---------------------------------------------------------
# Merge POST questionnaire
# ---------------------------------------------------------

post = ssq_lookup.rename(columns={
    "ID": "participant_id",
    "Questionnaire number": "post_questionnaire",
    "ssq_nausea": "post_ssq_nausea",
    "ssq_oculomotor": "post_ssq_oculomotor",
    "ssq_disorientation": "post_ssq_disorientation",
    "ssq_total": "post_ssq_total"
})

master = master.merge(
    post,
    on=["participant_id", "post_questionnaire"],
    how="left"
)


# ---------------------------------------------------------
# Calculate change in sickness
# ---------------------------------------------------------

master["delta_ssq_total"] = (
    master["post_ssq_total"] - master["pre_ssq_total"]
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

print("Sessions:", len(master))

score_cols = [
    "pre_ssq_total",
    "post_ssq_total",
    "delta_ssq_total"
]

print("\nMissing SSQ values:")
print(master[score_cols].isna().sum())

print("\nPre SSQ summary:")
print(master["pre_ssq_total"].describe())

print("\nPost SSQ summary:")
print(master["post_ssq_total"].describe())

print("\nDelta SSQ summary:")
print(master["delta_ssq_total"].describe())

print("\nFirst 10 mappings:")
print(
    master[
        [
            "participant_id",
            "session_number",
            "game_name",
            "pre_questionnaire",
            "post_questionnaire",
            "pre_ssq_total",
            "post_ssq_total",
            "delta_ssq_total"
        ]
    ].head(10).to_string(index=False)
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

output = PROCESSED_DIR / "questset_sessions_with_ssq.csv"
master.to_csv(output, index=False)

print(f"\nSaved: {output}")