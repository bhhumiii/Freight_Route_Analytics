import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

OD_FILE = os.path.join(DATA_DIR, "OD_FY2526_VIA_LIST.xlsx")
COORD_FILE = os.path.join(DATA_DIR, "smartrake_station_coords.xlsx")

# --------------------------------------------------
# Load datasets
# --------------------------------------------------

print("Loading OD dataset...")
df = pd.read_excel(OD_FILE)
print(f"Rows loaded: {len(df):,}")

print("Loading station coordinates...")
coords = pd.read_excel(COORD_FILE)

station_map = {}

for _, row in coords.iterrows():
    code = str(row["code"]).strip().upper()

    station_map[code] = {
        "name": row["name"],
        "lat": float(row["latitude"]),
        "lon": float(row["longitude"])
    }

print(f"Loaded coordinates for {len(station_map):,} stations")

# --------------------------------------------------
# Build adjacent station graph
# --------------------------------------------------

edge_stats = defaultdict(
    lambda: {
        "km": [],
        "hrs": [],
        "speed": []
    }
)

print("Building adjacent station graph...")

for idx, row in df.iterrows():

    if idx % 50000 == 0:
        print(f"Processed {idx:,}/{len(df):,} rows")

    route = str(row["travl_route_part"]).strip()

    if route == "" or route.lower() == "nan":
        continue

    stations = [
        s.strip().upper()
        for s in route.split(",")
        if s.strip()
    ]

    if len(stations) < 2:
        continue

    total_km = row["km"]
    total_hrs = row["avgtime_hrs"]

    if pd.isna(total_km) or pd.isna(total_hrs):
        continue

    segment_count = len(stations) - 1

    km_per_edge = total_km / segment_count
    hrs_per_edge = total_hrs / segment_count

    speed = 0
    if hrs_per_edge > 0:
        speed = km_per_edge / hrs_per_edge

    for i in range(segment_count):

        a = stations[i]
        b = stations[i + 1]

        edge_stats[(a, b)]["km"].append(km_per_edge)
        edge_stats[(a, b)]["hrs"].append(hrs_per_edge)
        edge_stats[(a, b)]["speed"].append(speed)

        edge_stats[(b, a)]["km"].append(km_per_edge)
        edge_stats[(b, a)]["hrs"].append(hrs_per_edge)
        edge_stats[(b, a)]["speed"].append(speed)

print(f"\nTotal unique directed edges: {len(edge_stats):,}")

# --------------------------------------------------
# Convert to graph
# --------------------------------------------------

print("Converting edge statistics into graph...")

graph = defaultdict(dict)

for (src, dst), stats in edge_stats.items():

    graph[src][dst] = {
        "km": float(np.mean(stats["km"])),
        "hrs": float(np.mean(stats["hrs"])),
        "speed": float(np.mean(stats["speed"])),
        "count": len(stats["km"])
    }

graph = dict(graph)

print(f"Graph nodes: {len(graph):,}")

# --------------------------------------------------
# Save files
# --------------------------------------------------

print("Saving graph.pkl...")

with open(os.path.join(DATA_DIR, "graph.pkl"), "wb") as f:
    pickle.dump(graph, f)

print("Saving station_map.pkl...")

with open(os.path.join(DATA_DIR, "station_map.pkl"), "wb") as f:
    pickle.dump(station_map, f)

print("Saving processed.pkl...")

with open(os.path.join(DATA_DIR, "processed.pkl"), "wb") as f:
    pickle.dump(df, f)

print("\n===================================")
print("Preprocessing completed successfully!")
print("===================================")
print(f"Stations       : {len(graph):,}")
print(f"Edges          : {len(edge_stats):,}")
print("Generated:")
print("  ✓ graph.pkl")
print("  ✓ station_map.pkl")
print("  ✓ processed.pkl")
