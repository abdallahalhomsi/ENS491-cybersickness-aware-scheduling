# Network Simulation Module

## Current Status
ns-3 and the vr-app module are fully installed and the baseline simulation is working. This folder contains the Python scheduler prototype, the scenario data, and the ns-3 C++ baseline.

## Realistic Network Parameters (from NeSt-VR & VR Gaming 2025)
- **Bitrate:** 100 Mbps shared link
- **Frame Rates:** 60 FPS (supported by vr-app module)
- **Latency:** 10 ms base delay
- **Jitter/Loss:** 0–20 ms jitter, 1% packet loss (NeSt-VR controlled impairment ranges)
- **Multi-user conditions:** 3 concurrent VR users (VR Gaming 2025 multi-user scenarios)

## Directory Structure
- `ns3/vr-edf-baseline.cc`: The C++ multi-user simulation with standard EDF queueing.
- `scheduler/simulator.py`: The Python proof-of-concept showing how the risk formula changes priority compared to EDF.
- `scheduler/risk_scores.json`: The interface file mapping user IDs to predicted risk scores (0 to 1).
- `scenarios/synthetic_beatsaber_ns3.csv`: A synthetic 60 FPS Beat Saber traffic trace used until the raw Questset data is available.

## Next Steps
- Replace synthetic trace with real Questset Beat Saber data.
- Implement the Risk-Aware scheduler C++ logic.