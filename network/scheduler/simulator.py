import json
import random
import matplotlib.pyplot as plt
import numpy as np

# --- 1. LOAD ML RISK SCORES ---
with open('risk_scores.json', 'r') as f:
    risk_scores = json.load(f)

# --- 2. SIMULATION PARAMETERS ---
NUM_FRAMES_PER_USER = 1000
FRAME_INTERVAL_MS = 14.0     
FRAME_DEADLINE_MS = 30.0     
SERVICE_TIME_MS = 5.5        # Goldilocks bottleneck
ALPHA = 1.0                  
EPSILON = 0.001

# Realistic Impairments (from NeSt-VR)
PACKET_LOSS_RATE = 0.01      # 1% loss
JITTER_MS = 2.0              # 2ms jitter

# --- 3. GENERATE VR TRAFFIC ---
packets = []
for user, risk in risk_scores.items():
    for i in range(NUM_FRAMES_PER_USER):
        base_arrival = i * FRAME_INTERVAL_MS 
        
        # Apply jitter
        arrival_time = base_arrival + random.uniform(-JITTER_MS, JITTER_MS)
        
        # Apply packet loss (skip creating the packet)
        if random.random() < PACKET_LOSS_RATE:
            continue
            
        deadline = base_arrival + FRAME_DEADLINE_MS
        packets.append({
            'id': f"{user}_{i}",
            'user': user,
            'arrival': arrival_time,
            'deadline': deadline,
            'risk': risk
        })

# Sort packets chronologically, breaking ties with the ID string
packets.sort(key=lambda x: (x['arrival'], x['id']))

# --- 4. THE SCHEDULERS ---
def run_simulation(scheduler_type):
    current_time = 0.0
    queue = []
    results = {user: {'on_time': 0, 'missed': 0, 'total': 0} for user in risk_scores}
    
    packet_idx = 0
    total_packets = len(packets)
    
    while packet_idx < total_packets or queue:
        # 1. Add all packets that have arrived by current_time to the queue
        while packet_idx < total_packets and packets[packet_idx]['arrival'] <= current_time:
            queue.append(packets[packet_idx])
            results[packets[packet_idx]['user']]['total'] += 1
            packet_idx += 1
            
        # 2. If queue is empty, jump time to the next arrival
        if not queue:
            current_time = packets[packet_idx]['arrival']
            continue
            
        # 3. Pick the best packet to serve based on scheduler
        best_packet = None
        best_priority = -1
        best_index = -1
        
        for i, q_packet in enumerate(queue):
            time_to_deadline = max(q_packet['deadline'] - current_time, EPSILON)
            
            if scheduler_type == 'EDF':
                priority = 1.0 / time_to_deadline
            elif scheduler_type == 'RISK_AWARE':
                priority = (1.0 + ALPHA * q_packet['risk']) / time_to_deadline
                
            if priority > best_priority:
                best_priority = priority
                best_packet = q_packet
                best_index = i
                
        # 4. Serve the packet
        queue.pop(best_index)
        
        # Check if it met the deadline
        if current_time <= best_packet['deadline']:
            results[best_packet['user']]['on_time'] += 1
        else:
            results[best_packet['user']]['missed'] += 1
            
        # 5. Advance time by the bottleneck service time
        current_time += SERVICE_TIME_MS

    return results

# --- 5. EVALUATION METRICS ---
def calculate_jain_fairness(results):
    x_i = []
    for user in results:
        total = results[user]['total']
        on_time = results[user]['on_time']
        x_i.append(on_time / total if total > 0 else 0)
    
    n = len(x_i)
    numerator = sum(x_i) ** 2
    denominator = n * sum([x ** 2 for x in x_i])
    return numerator / denominator if denominator != 0 else 0

# --- 6. RUN THE EXPERIMENTS ---
print("--- Running Baseline (EDF) ---")
edf_results = run_simulation('EDF')
for user, data in edf_results.items():
    print(f"{user}: Deadline Misses = {data['missed']}, On Time = {data['on_time']}")
print(f"EDF Jain Fairness Index: {calculate_jain_fairness(edf_results):.4f}\n")

print("--- Running Treatment (Risk-Aware) ---")
risk_results = run_simulation('RISK_AWARE')
for user, data in risk_results.items():
    print(f"{user}: Deadline Misses = {data['missed']}, On Time = {data['on_time']}")
print(f"Risk-Aware Jain Fairness Index: {calculate_jain_fairness(risk_results):.4f}\n")

# --- 7. PLOT THE RESULTS ---
users = ['User A (Risk 0.85)', 'User B (Risk 0.20)', 'User C (Risk 0.50)']
edf_misses = [edf_results['user_A']['missed'], edf_results['user_B']['missed'], edf_results['user_C']['missed']]
risk_misses = [risk_results['user_A']['missed'], risk_results['user_B']['missed'], risk_results['user_C']['missed']]

x = np.arange(len(users))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, edf_misses, width, label='Standard EDF', color='gray')
rects2 = ax.bar(x + width/2, risk_misses, width, label='Risk-Aware (alpha=1.0)', color='blue')

ax.set_ylabel('Deadline Misses (Lower is Better)')
ax.set_title('Network Performance: EDF vs Risk-Aware Scheduler')
ax.set_xticks(x)
ax.set_xticklabels(users)
ax.legend()

# Attach a label above each bar
ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

plt.tight_layout()
plt.show()