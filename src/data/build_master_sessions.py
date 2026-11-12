import pandas as pd
from pathlib import Path


RAW_DIR = Path("data/raw/Questset")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Load metadata
# ---------------------------------------------------------

sessions = pd.read_csv(RAW_DIR / "dataset_info.csv")
survey = pd.read_csv(RAW_DIR / "Initial survey.csv")
ssq = pd.read_csv(RAW_DIR / "SSQ.csv")


# ---------------------------------------------------------
# Create unique participant ID
# ---------------------------------------------------------

# User number alone is NOT unique.
# Example:
# group1_order1_user0
# group2_order1_user0
# are different participants.

sessions["participant_id"] = sessions.apply(
    lambda r: f"group{r['Group']}_order{r['Order']}_user{r['User']}",
    axis=1
)


# ---------------------------------------------------------
# Determine session order
# ---------------------------------------------------------
#
# Order = 1 -> slow game first, fast game second
# Order = 2 -> fast game first, slow game second
#
# SSQ sequence:
# Q1 = before session 1
# Q2 = after session 1
# Q3 = before session 2
# Q4 = after session 2
# ---------------------------------------------------------

def get_session_number(row):
    if row["Order"] == 1:
        return 1 if row["Game Speed"] == "slow" else 2
    else:
        return 1 if row["Game Speed"] == "fast" else 2


sessions["session_number"] = sessions.apply(
    get_session_number,
    axis=1
)


# ---------------------------------------------------------
# Map correct pre/post SSQ questionnaire numbers
# ---------------------------------------------------------

sessions["pre_questionnaire"] = sessions["session_number"].map({
    1: 1,
    2: 3
})

sessions["post_questionnaire"] = sessions["session_number"].map({
    1: 2,
    2: 4
})


# ---------------------------------------------------------
# Merge participant survey information
# ---------------------------------------------------------

survey = survey.rename(columns={
    "ID": "participant_id",
    "Previous VR experience (1-5)": "vr_experience",
    "Previous general gaming experience (1-5)": "gaming_experience"
})

sessions = sessions.merge(
    survey[
        [
            "participant_id",
            "Age",
            "Gender",
            "vr_experience",
            "gaming_experience"
        ]
    ],
    on="participant_id",
    how="left"
)


# ---------------------------------------------------------
# Clean column names
# ---------------------------------------------------------

sessions = sessions.rename(columns={
    "Group": "group",
    "Order": "order",
    "Game Speed": "game_speed",
    "Game Name": "game_name",
    "User": "user",
    "Traffic filepath": "traffic_filepath",
    "Movement filepath": "movement_filepath",
    "Duration (Traffic)": "traffic_duration",
    "Duration (Movement)": "movement_duration",
    "N Samples (Traffic)": "traffic_samples",
    "N Samples (Movement)": "movement_samples",
    "Age": "age",
    "Gender": "gender"
})


# ---------------------------------------------------------
# Arrange columns
# ---------------------------------------------------------

columns = [
    "participant_id",
    "group",
    "order",
    "user",
    "session_number",
    "game_speed",
    "game_name",
    "pre_questionnaire",
    "post_questionnaire",
    "age",
    "gender",
    "vr_experience",
    "gaming_experience",
    "traffic_filepath",
    "movement_filepath",
    "traffic_duration",
    "movement_duration",
    "traffic_samples",
    "movement_samples"
]

sessions = sessions[columns]


# ---------------------------------------------------------
# Sort
# ---------------------------------------------------------

sessions = sessions.sort_values(
    ["participant_id", "session_number"]
).reset_index(drop=True)


# ---------------------------------------------------------
# Validation checks
# ---------------------------------------------------------

print("Participants:", sessions["participant_id"].nunique())
print("Sessions:", len(sessions))

print(
    "Sessions per participant:",
    sessions.groupby("participant_id").size().value_counts().to_dict()
)

print(
    "Missing survey values:",
    sessions[
        ["age", "gender", "vr_experience", "gaming_experience"]
    ].isna().sum().to_dict()
)

print("\nGame counts:")
print(sessions["game_name"].value_counts())

print("\nSpeed counts:")
print(sessions["game_speed"].value_counts())

print("\nQuestionnaire mapping:")
print(
    sessions[
        [
            "participant_id",
            "order",
            "session_number",
            "game_speed",
            "game_name",
            "pre_questionnaire",
            "post_questionnaire"
        ]
    ].head(12).to_string(index=False)
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

output_path = OUT_DIR / "questset_master_sessions.csv"
sessions.to_csv(output_path, index=False)

print(f"\nSaved: {output_path}")