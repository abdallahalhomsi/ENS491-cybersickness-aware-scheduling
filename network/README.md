# Network Simulation Module (ns-3 + vr-app)

## Current Status: Fully Operational & Risk-Aware

The ns-3 environment is fully built and running. The simulation now successfully ingests ML predictions from `ml_risk_scores.json` and applies the Risk-Aware scheduling formula alongside the standard EDF baseline.

## Realistic Network Parameters (NeSt-VR & VR Gaming 2025)

- **Bandwidth:** 100 Mbps shared link
- **Packet Loss:** 1% (NeSt-VR controlled impairment)
- **Delay:** 10 ms base latency
- **Frame Rate:** Real Beat Saber variable frame sizes (1 KB to 600 KB+ bursts)

## Multi-User Setup & Results

The simulation runs 3 concurrent VR users competing for bandwidth. We tested multiple risk weights (\(\alpha\)):

- **`edf_results.txt` (\(\alpha = 0\))**: All users receive equal priority weight (1.0).
- **`risk_results.txt` (\(\alpha = 1.0\))**: The scheduler prioritises vulnerable users (User 1 risk 0.85 gets priority 1.85; User 2 risk 0.20 gets priority 1.20).
- **`risk_results_alpha_0.5.txt`** and **`risk_results_alpha_2.0.txt`**: Show how aggressive or conservative the scheduler behaves at different weights.

When fed with Abdallah's real traffic, the baseline EDF queue processed **583 MB of real VR data** (1,181 bursts), correctly dropping packets under congestion to protect deadlines.

## ML Interface

The file `ml_risk_scores.json` maps `user_id` to `risk_score` (0.0 to 1.0). The C++ code reads this at the start of the simulation and calculates the priority weight for each user using:

\[
P_i = \frac{1 + \alpha R_i}{\max(D_i - t, \epsilon)}
\]

## Directory Structure

- `ns3/` - Contains the C++ code (`vr-risk-scheduler.cc`).
- `scenarios/` - Contains the synthetic and real Beat Saber traces.
- `scheduler/` - Contains the Python proof-of-concept and the JSON interface.
- `results/` - Contains terminal logs for EDF and all tested Risk-Aware alpha values.

---

# Instructions for the Team

## 1. Prerequisites (If You Are on a New Machine)

If you are setting this up from scratch on WSL/Ubuntu, you need to install the required tools and download the ns-3 source code.

### Install the Build Tools

```bash
sudo apt update
sudo apt install build-essential cmake python3 git gcc-11 g++-11 -y
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 --slave /usr/bin/g++ g++ /usr/bin/g++-11
```

### Clone ns-3 and the VR App Module

> **Note:** Only clone these if you do not already have the `/mnt/d/ns3_project/ns-3-dev` folder. If you are using Manal's pre-built D: drive folder, skip this step.

```bash
git clone https://gitlab.com/nsnam/ns-3-dev ns-3-dev
git clone https://github.com/signetlabdei/ns-3-vr-app ns-3-dev/contrib/vr-app
```

## 2. How to Run the Simulation

> **IMPORTANT:** Do not use `./ns3 run`. The ns-3 script checks for file changes and may trigger a massive rebuild if timestamps are off. **Always use the compiled binary directly.**

### Go to the ns-3 Directory

```bash
cd /mnt/d/ns3_project/ns-3-dev
```

### Run the Standard Example

```bash
./build/contrib/vr-app/examples/ns3-dev-vr-application-example-default
```

### Run the Risk-Aware Scheduler

```bash
# Run EDF Baseline (alpha = 0)
./build/contrib/vr-app/examples/ns3-dev-vr-risk-scheduler-default --alpha=0 > edf_results.txt

# Run Risk-Aware (alpha = 1.0)
./build/contrib/vr-app/examples/ns3-dev-vr-risk-scheduler-default --alpha=1.0 > risk_results.txt
```

## 3. To Change the ML Predictions

When Abdallah provides the final risk scores, simply overwrite the existing `ml_risk_scores.json` in the root `ns-3-dev` folder with the real data and run the Risk-Aware scheduler command.

**No rebuild is needed.**

## 4. What NOT to Do

1. **DO NOT** delete the `build/` folder.
2. **DO NOT** delete the `cmake-cache/` folder.
3. **DO NOT** run `./ns3 build` or `./ns3 configure` unless you are prepared for a potentially very long rebuild.