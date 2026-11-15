# Network Simulation Module (ns-3 + vr-app)

## Current Status: Fully Operational

The ns-3 baseline environment is fully built and successfully running on Abdallah's real Questset network traffic.

## Realistic Network Parameters (NeSt-VR & VR Gaming 2025)

- **Bandwidth:** 100 Mbps shared link
- **Packet Loss:** 1% (NeSt-VR controlled impairment)
- **Delay:** 10 ms base latency
- **Frame Rate:** Real Beat Saber variable frame sizes (ranging from 1 KB to 600 KB+ bursts)

## Multi-User Setup & EDF Baseline

The baseline scenario (`vr-edf-baseline.cc`) simulates 3 concurrent VR users competing for bandwidth. The router uses standard Earliest Deadline First (EDF) queueing (`PfifoFastQueueDisc`).

When fed with Abdallah's real traffic, the simulation processed **583 MB of real VR data** (1,181 bursts sent). Because the default 100 Mbps link was constrained, the EDF queue correctly dropped packets to protect deadlines, demonstrating realistic network congestion.

## ML Interface

The file `ml_risk_scores.json` maps `user_id` to `risk_score` (0.0 to 1.0). The network side is ready to ingest predictions at the 60-second mark.

## Directory Structure

- `ns3/`: Contains the C++ baseline code (`vr-edf-baseline.cc`).
- `scenarios/`: Contains the synthetic Beat Saber trace.
- `scheduler/`: Contains the Python proof-of-concept and the JSON interface.
- `results/`: Contains the terminal output logs of the real-data EDF simulation.

---

# Instructions for the Team

## 1. Prerequisites (If You Are on a New Machine)

If you are setting this up from scratch on WSL/Ubuntu, you need to install the required tools and download the ns-3 source code.

### Install the Build Tools

```bash
sudo apt update
sudo apt install build-essential cmake python3 git gcc-11 g++-11 -y

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
  --slave /usr/bin/g++ g++ /usr/bin/g++-11

Clone ns-3 and the VR App Module

Note: Only clone these repositories if you do not already have the /mnt/d/ns3_project/ns-3-dev folder. If you are using Manal's pre-built D: drive folder, skip this step.

git clone https://gitlab.com/nsnam/ns-3-dev ns-3-dev
git clone https://github.com/signetlabdei/ns-3-vr-app \
  ns-3-dev/contrib/vr-app

2. How to Run the Simulation (Instant)

IMPORTANT: Do not use ./ns3 run. The ns-3 script checks for file changes and may trigger a massive rebuild if timestamps are off. Always use the compiled binary directly.

Go to the ns-3 Directory
cd /mnt/d/ns3_project/ns-3-dev

Run the Pre-Compiled Binary
./build/contrib/vr-app/examples/ns3-dev-vr-application-example-default

3. To Change the Traffic Data

If you want to test a different trace (e.g., another user's Beat Saber data):

Convert the raw CSV to ns-3 format using convert_real_data.py.
Copy the new file over the existing one in both the source and build folders:
cp your_new_trace.csv \
  contrib/vr-app/model/BurstGeneratorTraces/vp_40mbps_30fps.csv

cp your_new_trace.csv \
  build/contrib/vr-app/model/BurstGeneratorTraces/vp_40mbps_30fps.csv

Run the binary again. No rebuild is required.
4. What NOT to Do
DO NOT delete the build/ folder.
DO NOT delete the cmake-cache/ folder.
DO NOT run ./ns3 build or ./ns3 configure unless you are prepared for a potentially very long rebuild.
