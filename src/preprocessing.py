"""
preprocessing.py
Loads the raw rake-cycle CSV, cleans data, detects/handles outliers,
engineers features for the ML models, and builds a station-to-station
graph for path finding.
"""
import pandas as pd
import numpy as np
import pickle, os

import config as C

DATA_DIR    = C.DATA_DIR
PROC_PKL    = C.PROC_PKL
GRAPH_PKL   = C.GRAPH_PKL
SMAP_PKL    = C.SMAP_PKL
OUTLIER_PKL = os.path.join(DATA_DIR, "outlier_report.pkl")
COORDS_CSV = os.path.join(DATA_DIR, "smartrake_station_coords.csv")

# Columns whose distributions we screen for outliers (numeric, physically meaningful)
OUTLIER_COLS = [
    "ldng_uldg_km",
    "ldng_uldg_hor",
    "circuit_speed",
    "ldng_uldg_speed",
    "actlwght",
    "chblwght",
    "load_units"
]


# ── helpers ──────────────────────────────────────────────────────────────────
def parse_hhmm(t):
    """'408:52' -> 408.867 hours. Returns NaN on malformed input."""
    try:
        h, m = str(t).split(":")
        return int(h) + int(m) / 60
    except (ValueError, AttributeError):
        return np.nan


def detect_outliers(series, method="iqr", iqr_k=1.75, z_thresh=3.0):
    """
    Return a boolean mask (True = outlier) for a numeric series.
      method='iqr'    -> Tukey fences  [Q1 - k*IQR, Q3 + k*IQR]
      method='zscore' -> |z| > z_thresh
    NaNs are treated as non-outliers (handled separately).
    """
    s = pd.to_numeric(series, errors="coerce")
    if method == "zscore":
        mu, sd = s.mean(), s.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return pd.Series(False, index=series.index)
        z = (s - mu) / sd
        return z.abs() > z_thresh
    # default: IQR
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - iqr_k * iqr, q3 + iqr_k * iqr
    return (s < lower) | (s > upper)


def outlier_summary(df, cols=OUTLIER_COLS, method="iqr"):
    """Build a per-column outlier report (bounds, counts, pct) for diagnostics."""
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        mask = detect_outliers(s, method=method)
        rows.append({
            "column": c,
            "method": method,
            "q1": round(float(q1), 3),
            "q3": round(float(q3), 3),
            "iqr": round(float(iqr), 3),
            "lower_fence": round(float(q1 - 1.5 * iqr), 3),
            "upper_fence": round(float(q3 + 1.5 * iqr), 3),
            "mean": round(float(s.mean()), 3),
            "std": round(float(s.std(ddof=0)), 3),
            "n_outliers": int(mask.sum()),
            "pct_outliers": round(100 * mask.mean(), 3),
        })
    return pd.DataFrame(rows)


def load_and_clean(outlier_method="iqr", cap_outliers=True):
    """
    Load + clean the raw CSV. Outliers in the core numeric columns are
    detected (IQR by default) and *capped* to the fences (winsorised)
    rather than dropped, so the graph keeps its connectivity while the
    ML target distribution is de-noised. Set cap_outliers=False to skip.
    """
    raw_csv = C.resolve_raw_csv()
    print(f"Loading CSV ({os.path.basename(raw_csv)}) ...")
    df = pd.read_csv(raw_csv, low_memory=False)
    print(f"  Raw rows: {len(df):,}")

    # ── parse circuit time to hours ──────────────────────────────────────────
    df["circuit_hrs"] = df["circuittime"].apply(parse_hhmm)

    # ── keep only rows with the core columns present and positive ────────────
    df = df.dropna(subset=["ldngsttn", "uldg_sttn","ldng_uldg_km", "circuit_hrs", "circuit_speed"])
    extra_edges = df[
    [
        "uldg_sttn",
        "nextldng",
        "uldg_ldng_kms",
        "uldg_ldng_hrs"
    ]].dropna()

    extra_edges = extra_edges.rename(
        columns={
            "uldg_sttn":"ldngsttn",
            "nextldng":"uldg_sttn",
            "uldg_ldng_kms":"ldng_uldg_km",
            "uldg_ldng_hrs":"ldng_uldg_hor"
        }
        )

    extra_edges["circuit_speed"] = (
        extra_edges["ldng_uldg_km"] /
        extra_edges["ldng_uldg_hor"]
    )

    df = pd.concat(
        [df,extra_edges],
        ignore_index=True
    )
    df = df[df["ldng_uldg_km"] > 0]
    df = df[df["circuit_hrs"]  > 0]
    df = df[df["circuit_speed"] > 0]
    print(f"  After clean: {len(df):,}")

    # avg speed loading->unloading (also an outlier-screened column)
    df["ldng_uldg_speed"] = df["ldng_uldg_km"] / df["ldng_uldg_hor"].replace(0, np.nan)

    # ── outlier detection + handling ─────────────────────────────────────────
    report = outlier_summary(df, OUTLIER_COLS, method=outlier_method)
    print(f"  Outlier report ({outlier_method}):")
    for _, r in report.iterrows():
        print(f"    {r['column']:16s} {r['n_outliers']:>7,} rows "
            f"({r['pct_outliers']:.2f}%) outside [{r['lower_fence']}, {r['upper_fence']}]")

    if cap_outliers:
        capped_total = 0
        for c in OUTLIER_COLS:
            if c not in df.columns:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            before = ((s < lo) | (s > hi)).sum()
            df[c] = s.clip(lower=lo, upper=hi)
            capped_total += int(before)
        print(f"  Winsorised {capped_total:,} outlier values to IQR fences "
              f"across {len(OUTLIER_COLS)} columns")

    # ======================================================
# Station Map with Coordinates
# ======================================================

    df["ldngsttn"] = df["ldngsttn"].str.strip()
    df["uldg_sttn"] = df["uldg_sttn"].str.strip()

    coords = pd.read_csv(COORDS_CSV)

    coords["code"] = coords["code"].astype(str).str.strip()

    coord_lookup = coords.set_index("code").to_dict("index")

    smap = {}

# Loading stations
    for _, r in df[["ldngsttn", "ldngname"]].drop_duplicates().iterrows():

        code = str(r["ldngsttn"]).strip()

        if code in coord_lookup:

            smap[code] = {

                "name": str(r["ldngname"]).strip(),

                "lat": float(coord_lookup[code]["latitude"]),

                "lon": float(coord_lookup[code]["longitude"])

            }

# Unloading stations
    for _, r in df[["uldg_sttn", "uldgfullname"]].drop_duplicates().iterrows():

        code = str(r["uldg_sttn"]).strip()

        if code in coord_lookup:

            smap[code] = {

                "name": str(r["uldgfullname"]).strip(),

                "lat": float(coord_lookup[code]["latitude"]),

                "lon": float(coord_lookup[code]["longitude"])

        }

    print(f"Matched coordinates for {len(smap)} stations.")

    # ── feature engineering ──────────────────────────────────────────────────
    # zone match flag
    df["same_zone"] = (df["ldngzone"] == df["uldgzone"]).astype(int)

    # distance buckets — bins MUST match the inference-time bucketing in
    # pathfinder._dist_bucket() to avoid train/serve skew.
    df["dist_bucket"] = pd.cut(
        df["ldng_uldg_km"],
        bins=C.DIST_BUCKET_BINS, labels=C.DIST_BUCKET_LABELS,
    ).astype(float)

    # log-distance feature
    df["log_dist_km"] = np.log1p(df["ldng_uldg_km"])
    # =====================================================
# Additional Feature Engineering
# =====================================================

# Average speed
    df["avg_speed"] = (
        df["ldng_uldg_km"] /
        (df["ldng_uldg_hor"] + 1e-6)
    )

# Weight carried per wagon
    df["weight_per_unit"] = (
        df["actlwght"] /
        (df["load_units"] + 1)
)

# Wagon utilization
    df["loading_efficiency"] = (
    df["actlwght"] /
    (df["chblwght"] + 1)
)

# Long route indicator
    df["long_route"] = (
        df["ldng_uldg_km"] > 700
        ).astype(int)

# Heavy train indicator
    df["heavy_train"] = (
    df["actlwght"] > 3500
    ).astype(int)

# Time per km
    df["hrs_per_km"] = (
        df["ldng_uldg_hor"] /
        (df["ldng_uldg_km"] + 1)
)
    # encode categoricals
    for c in C.CAT_COLS:
        df[c] = df[c].fillna("UNKNOWN").astype("category")

    return df, smap, report


def build_graph(df):
    """
    Weighted directed graph:
      node = station code
      edge = (src -> dst) with median km / hrs / speed and a composite weight.
    Aggregated per (ldngsttn, uldg_sttn) pair using medians.
    """
    print("Building route graph ...")
    grp = (
        df.groupby(["ldngsttn", "uldg_sttn"])
        .agg(
            median_km    = ("ldng_uldg_km",  "median"),
            median_hrs   = ("ldng_uldg_hor", "median"),
            median_speed = ("circuit_speed", "median"),
            count        = ("ldng_uldg_km",  "count"),
        )
        .reset_index()
    )

    # composite score (lower = better): 0.5*norm_km + 0.5*norm_hrs
    def _norm(col):
        lo, hi = col.min(), col.max()
        return (col - lo) / (hi - lo + 1e-9)

    grp["norm_km"]  = _norm(grp["median_km"])
    grp["norm_hrs"] = _norm(grp["median_hrs"])
    grp["weight"] = (
    0.35 * grp["norm_km"] +
    0.65 * grp["norm_hrs"]
    )

    graph = {}

    for r in grp.itertuples(index=False):

        src = str(r.ldngsttn).strip()
        dst = str(r.uldg_sttn).strip()

        graph.setdefault(src, {})
        graph.setdefault(dst, {})       # prevents dead-end nodes

        [src][dst] = {
            "km": float(r.median_km),
            "hrs": float(r.median_hrs),
            "speed": float(r.median_speed),
            "weight": float(r.weight),
            "count": int(r.count),
            "from_name": smap.get(src, {}).get("name", src),
            "to_name": smap.get(dst, {}).get("name", dst)
        }

    print(f"  Nodes: {len(graph):,} | Edges: {sum(len(v) for v in graph.values()):,}")
    return graph


def run(outlier_method="iqr", cap_outliers=True):
    df, smap, report = load_and_clean(outlier_method=outlier_method,
                                      cap_outliers=cap_outliers)
    graph = build_graph(df)

    with open(PROC_PKL,    "wb") as f: pickle.dump(df,     f, protocol=5)
    with open(GRAPH_PKL,   "wb") as f: pickle.dump(graph,  f, protocol=5)
    with open(SMAP_PKL,    "wb") as f: pickle.dump(smap,   f, protocol=5)
    with open(OUTLIER_PKL, "wb") as f: pickle.dump(report, f, protocol=5)
    print("Saved processed.pkl, graph.pkl, station_map.pkl, outlier_report.pkl")
    return df, graph, smap


if __name__ == "__main__":
    run()
