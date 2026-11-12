import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw/Questset/traffic")
OUT_DIR = Path("data/processed")

WINDOWS = {
    "1min": 60,
    "3min": 180,
    "5min": 300
}


def stats(series, prefix):
    series = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()

    if len(series) == 0:
        return {}

    return {
        f"{prefix}_mean": series.mean(),
        f"{prefix}_std": series.std(),
        f"{prefix}_median": series.median(),
        f"{prefix}_p25": series.quantile(0.25),
        f"{prefix}_p75": series.quantile(0.75),
        f"{prefix}_p95": series.quantile(0.95),
        f"{prefix}_max": series.max()
    }


rows = []

for path in sorted(RAW_DIR.glob("*.csv")):

    print("Processing:", path.name)

    df = pd.read_csv(path)

    for window_name, seconds in WINDOWS.items():

        w = df[df["time"] <= seconds].copy()

        if len(w) == 0:
            continue

        duration = w["time"].max() - w["time"].min()

        if duration <= 0:
            continue

        dl = w[w["direction"] == "DL"]
        ul = w[w["direction"] == "UL"]

        row = {
            "file": path.name,
            "window": window_name,
            "window_seconds": seconds,
            "packet_count_total": len(w),
            "packet_count_dl": len(dl),
            "packet_count_ul": len(ul),
            "packet_rate_total": len(w) / duration,
            "packet_rate_dl": len(dl) / duration,
            "packet_rate_ul": len(ul) / duration,
            "bytes_total": w["size"].sum(),
            "bytes_dl": dl["size"].sum(),
            "bytes_ul": ul["size"].sum(),
            "bitrate_mbps_total": w["size"].sum() * 8 / duration / 1e6,
            "bitrate_mbps_dl": dl["size"].sum() * 8 / duration / 1e6,
            "bitrate_mbps_ul": ul["size"].sum() * 8 / duration / 1e6,
        }

        if len(ul) > 0:
            row["dl_ul_packet_ratio"] = len(dl) / len(ul)
            row["dl_ul_byte_ratio"] = dl["size"].sum() / max(ul["size"].sum(), 1)

        row.update(stats(w["size"], "packet_size"))

        times = w["time"].to_numpy()

        if len(times) > 1:
            interarrival = np.diff(times)
            row.update(stats(interarrival, "interarrival"))

            mean_iat = np.mean(interarrival)
            std_iat = np.std(interarrival)

            if mean_iat > 0:
                row["interarrival_cv"] = std_iat / mean_iat

        rows.append(row)


out = pd.DataFrame(rows)

output_path = OUT_DIR / "questset_traffic_sample_features.csv"
out.to_csv(output_path, index=False)

print("\nRows:", len(out))
print("\nFeatures:")
print(out.to_string(index=False))
print(f"\nSaved: {output_path}")