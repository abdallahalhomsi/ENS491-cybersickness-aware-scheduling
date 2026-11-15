import csv

INPUT_FILE = '/home/huawei/data.csv'
OUTPUT_FILE = 'real_beatsaber_ns3.csv'
FRAME_THRESHOLD = 0.005 # 5 milliseconds

print(f"Reading {INPUT_FILE}...")

bursts = []
current_burst_size = 0
last_packet_time = None
last_burst_time = None

with open(INPUT_FILE, 'r') as f:
    reader = csv.reader(f)
    header = next(reader) # Skips the "time,size,direction" header
    print(f"Header: {header}")
    
    for row in reader:
        try:
            pkt_time = float(row[0])
            pkt_size = int(row[1])
            direction = row[2]
        except (ValueError, IndexError):
            continue
            
        # Only process Downlink (DL) packets
        if direction != 'DL':
            continue
            
        if last_packet_time is None:
            last_packet_time = pkt_time
            last_burst_time = pkt_time
            current_burst_size = pkt_size
            continue
            
        time_diff = pkt_time - last_packet_time
        
        # If packet arrives within 5ms of the previous one, group it into the same frame
        if time_diff < FRAME_THRESHOLD:
            current_burst_size += pkt_size
        else:
            # Save the previous frame (burst) and start a new one
            inter_arrival = last_packet_time - last_burst_time
            bursts.append((current_burst_size, inter_arrival))
            
            current_burst_size = pkt_size
            last_burst_time = pkt_time
            
        last_packet_time = pkt_time

# Save the final burst
if current_burst_size > 0:
    inter_arrival = last_packet_time - last_burst_time
    bursts.append((current_burst_size, inter_arrival))

print(f"Writing {len(bursts)} bursts to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w') as f:
    for size, interval in bursts:
        f.write(f"{size},{interval}\n")

print("Done! Successfully converted real Questset data.")
