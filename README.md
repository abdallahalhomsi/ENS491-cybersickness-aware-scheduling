# ENS491 Cybersickness-Aware Scheduling

Sabancı University ENS491 graduation project.

## Project Goal

This project investigates whether early cybersickness risk prediction can improve network resource allocation for VR users.

The system combines:

- Early cybersickness prediction from Questset
- Human-state validation using external VR datasets
- Network simulation using ns-3 and vr-app
- Risk-aware scheduling compared with standard EDF

## Main Research Pipeline

1. Extract early-session features from Questset
2. Predict post-session cybersickness risk
3. Convert predicted sickness into a normalized risk score
4. Feed the risk score into a network scheduler
5. Compare standard EDF with risk-aware EDF

## Main Data Sources

- Questset
- VRWalking
- Dense-FMS
- MazeSick
- VR Gaming
- NeSt-VR
- BRAVR
- ns-3 vr-app

Raw datasets are not stored in this repository.

## Repository Structure

- `data/` — processed data and data documentation
- `notebooks/` — exploration and experimental notebooks
- `src/` — preprocessing, features, models, and evaluation code
- `network/` — ns-3 simulations and scheduling code
- `results/` — figures and result tables
- `literature/` — literature review notes
- `paper/` — manuscript notes and figures