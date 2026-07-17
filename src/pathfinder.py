"""
pathfinder.py
Dijkstra-based optimal path finder over the rake-movement graph.
Three objectives: minimum distance, minimum ML-predicted time, and a
balanced composite. ML models predict circuit_speed per edge to drive
the time estimate.

Commodity & rake-type support: find_optimal_path now accepts `commodity`
and `rake_type` parameters. When they differ from the defaults (COAL / BOXN),
a fresh batched prediction is run for the new combo and cached in
self._edge_speed_cache so subsequent queries with the same combo are O(1).

Performance note: the default (COAL/BOXN) edge speeds are precomputed at
init. Dijkstra does O(1) lookups into whichever speed table is active for
the current query.

Balanced-mode normalisation: both distance (km) and time (hrs) components
are divided by their respective graph-wide maxima before weighting, so
both live in [0, 1] and the 0.6 / 0.4 split is meaningful.
"""
import heapq
import pickle
import numpy as np
import pandas as pd

import config as C
from config import dist_bucket as _dist_bucket

GRAPH_PKL  = C.GRAPH_PKL
SMAP_PKL   = C.SMAP_PKL
MODELS_PKL = C.MODELS_PKL

DEFAULT_COMMODITY = "COAL"
DEFAULT_RAKE_TYPE = "BOXN"


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class RailPathFinder:
    def __init__(self):
        print("Loading graph, station map, models ...")
        self.graph  = _load(GRAPH_PKL)
        self.smap   = _load(SMAP_PKL)
        self.bundle = _load(MODELS_PKL)

        self.encoders  = self.bundle["encoders"]
        self.feat_cols = self.bundle["feature_cols"]
        self.cat_cols  = self.bundle["cat_cols"]

        self.models = {
            "XGBoost":  self.bundle["xgboost"],
            "LightGBM": self.bundle["lightgbm"],
            "CatBoost": self.bundle["catboost"],
        }
        self.best_name  = self.bundle["best_model"]
        self.best_model = self.models[self.best_name]

        # Precompute the full node set once so find_optimal_path
        # does an O(1) existence check instead of rebuilding on every query.
        self._all_nodes: set = set(self.graph.keys()) | {
            v for nbrs in self.graph.values() for v in nbrs
        }

        # Pre-encode constant categorical defaults once.
        self._defaults_cat = {
            "ldngzone":     "NR",
            "uldgzone":     "NR",
            "ldngfromdvsn": "DLI",
            "uldgdvsn":     "DLI",
            "raketype":     DEFAULT_RAKE_TYPE,
            "grupcmdt":     DEFAULT_COMMODITY,
            "ldng_stat":    "UP",
            "uldg_state":   "UP",
        }
        self._cat_default_codes: dict = self._encode_cat_codes(self._defaults_cat)

        # Batch-precompute ML speeds for every unique (km, hrs) in the graph
        # using the default commodity / rake-type.
        self._edge_speed: dict = self._batch_predict_speeds(self._cat_default_codes)

        # Cache for non-default (commodity, rake_type) combos — populated lazily.
        self._edge_speed_cache: dict = {}

        # Balanced-mode normalisation denominators (per model).
        self._bal_norm: dict = self._compute_balance_norms(self._edge_speed)

        print(
            f"  Active model : {self.best_name}\n"
            f"  Edge cache   : {sum(len(v) for v in self._edge_speed.values())} "
            f"entries ({len(self.models)} models)\n"
            f"  Graph nodes  : {len(self._all_nodes)}"
        )

    # ── option lists for UI dropdowns ────────────────────────────────────────

    def get_valid_commodities(self) -> list[str]:
        return sorted(self.encoders["grupcmdt"].classes_.tolist())

    def get_valid_rake_types(self) -> list[str]:
        return sorted(self.encoders["raketype"].classes_.tolist())

    # ── categorical encoding helpers ─────────────────────────────────────────

    def _encode_cat_codes(self, overrides: dict) -> dict:
        """
        Build a {col: int_code} dict from the label encoders.
        `overrides` provides values for specific columns; columns absent from
        overrides fall back to the first class in the encoder.
        """
        codes = {}
        for c in self.cat_cols:
            le  = self.encoders[c]
            val = overrides.get(c, str(le.classes_[0]))
            if val not in set(le.classes_):
                val = le.classes_[0]
            codes[c] = int(le.transform([val])[0])
        return codes

    def _cat_codes_for(self, commodity: str, rake_type: str) -> dict:
        """Return cat_codes with only commodity and rake_type overridden."""
        overrides = dict(self._defaults_cat)
        overrides["grupcmdt"] = commodity
        overrides["raketype"] = rake_type
        return self._encode_cat_codes(overrides)

    # ── precomputation ────────────────────────────────────────────────────────

    def _batch_predict_speeds(self, cat_codes: dict) -> dict:
        """
        Return {model_name: {(km, hrs): predicted_speed}} for all graph edges,
        using the supplied categorical codes for the fixed feature columns.
        """
        pairs = sorted({
            (round(float(a["km"]), 1), round(float(a["hrs"]), 2))
            for nbrs in self.graph.values()
            for a in nbrs.values()
        })
        if not pairs:
            return {name: {} for name in self.models}

        kms      = np.array([p[0] for p in pairs], dtype=np.float64)
        hrs      = np.array([p[1] for p in pairs], dtype=np.float64)
        hrs_safe = np.clip(hrs, 0.01, None)

        feat = {
            "ldng_uldg_km":    kms,
            "log_dist_km":     np.log1p(kms),
            "ldng_uldg_hor":   hrs,
            "ldng_uldg_speed": kms / hrs_safe,
            "same_zone":       np.zeros_like(kms),
            "dist_bucket":     np.array([_dist_bucket(k) for k in kms]),
            "actlwght":        np.full_like(kms, 2500.0),
            "chblwght":        np.full_like(kms, 2500.0),
            "load_units":      np.full_like(kms, 40.0),
        }
        for c, code in cat_codes.items():
            feat[c] = np.full_like(kms, code)

        X   = pd.DataFrame(feat, columns=self.feat_cols).astype(np.float32)
        out = {}
        for name, model in self.models.items():
            preds = np.clip(np.asarray(model.predict(X), dtype=float), 1.0, None)
            out[name] = dict(zip(pairs, preds))
        return out

    def _get_edge_speeds(self, commodity: str, rake_type: str) -> dict:
        """
        Return the edge-speed table for the given (commodity, rake_type) combo.
        Uses the precomputed default table for COAL/BOXN; computes and caches
        a fresh table for any other combo (one batch predict per new combo).
        """
        if commodity == DEFAULT_COMMODITY and rake_type == DEFAULT_RAKE_TYPE:
            return self._edge_speed

        cache_key = (commodity, rake_type)
        if cache_key not in self._edge_speed_cache:
            print(f"  Computing edge speeds for commodity={commodity}, rake={rake_type} …")
            cat_codes = self._cat_codes_for(commodity, rake_type)
            self._edge_speed_cache[cache_key] = self._batch_predict_speeds(cat_codes)
        return self._edge_speed_cache[cache_key]

    def _compute_balance_norms(self, edge_speeds: dict) -> dict:
        """
        For each model compute max_km and max_time_hrs across all graph edges
        so the balanced Dijkstra can normalise both components to [0, 1].
        """
        norms = {}
        for name, speed_table in edge_speeds.items():
            max_km   = 1.0
            max_time = 1.0
            for (km, _hrs), speed in speed_table.items():
                max_km   = max(max_km, km)
                max_time = max(max_time, km / max(speed, 1.0))
            norms[name] = {"max_km": max_km, "max_time": max_time}
        return norms

    # ── coordinate helper ─────────────────────────────────────────────────────

    def _build_coords(self, path: list) -> list:
        coords = []
        for station in path:
            info = self.smap.get(station)
            if not info:
                continue
            coords.append((float(info["lat"]), float(info["lon"]), info["name"]))
        return coords

    # ── station helpers ───────────────────────────────────────────────────────

    def station_name(self, code: str) -> str:
        data = self.smap.get(code)
        if isinstance(data, dict):
            return data["name"]
        return code

    def search_stations(self, query: str) -> list:
        q       = query.strip().upper()
        results = []
        for code, info in self.smap.items():
            name = info["name"]
            if q in code.upper() or q in name.upper():
                results.append((code, name))
        results.sort(key=lambda x: (not x[0].startswith(q), x[0]))
        return results[:20]

    # ── ML speed prediction ───────────────────────────────────────────────────

    def _predict_speed(
        self,
        km: float,
        hrs: float,
        model_name: str,
        edge_speeds: dict,
    ) -> float:
        """
        O(1) lookup in the active edge-speed table.
        Falls back to a live single-row prediction for any (km, hrs) pair not
        found in the table (should not occur for normal graph edges).
        """
        key   = (round(float(km), 1), round(float(hrs), 2))
        table = edge_speeds.get(model_name, {})
        if key in table:
            return table[key]
        return self._predict_speed_single(km, hrs, model_name, edge_speeds)

    def _predict_speed_single(
        self,
        km: float,
        hrs: float,
        model_name: str,
        edge_speeds: dict,
    ) -> float:
        """
        Safety-net fallback for (km, hrs) pairs absent from the precomputed table.
        Not decorated with @lru_cache to avoid keeping 'self' (with all ML models)
        alive in the cache forever.
        """
        model    = self.models.get(model_name, self.best_model)
        hrs_safe = max(hrs, 0.01)

        # Reconstruct cat_codes from active edge_speeds table identity.
        # Since we can't reverse-look-up cat codes from the table, use defaults.
        cat_codes = self._cat_default_codes

        num = {
            "ldng_uldg_km":    km,
            "log_dist_km":     float(np.log1p(km)),
            "ldng_uldg_hor":   hrs,
            "ldng_uldg_speed": km / hrs_safe,
            "same_zone":       0,
            "dist_bucket":     _dist_bucket(km),
            "actlwght":        2500.0,
            "chblwght":        2500.0,
            "load_units":      40.0,
        }
        row   = {**num, **cat_codes}
        X     = pd.DataFrame([row], columns=self.feat_cols).astype(np.float32)
        speed = float(model.predict(X)[0])
        return max(speed, 1.0)

    # ── Dijkstra ──────────────────────────────────────────────────────────────

    def _dijkstra(
        self,
        source: str,
        target: str,
        mode: str,
        model_name: str,
        edge_speeds: dict,
        bal_norm: dict,
    ):
        if source not in self.graph:
            return None, None, None

        INF      = float("inf")
        norm     = bal_norm.get(model_name, {"max_km": 1.0, "max_time": 1.0})
        max_km   = norm["max_km"]
        max_time = norm["max_time"]

        dist = {source: 0.0}
        prev = {}
        heap = [(0.0, source)]

        while heap:
            cost, u = heapq.heappop(heap)
            if cost > dist.get(u, INF):
                continue
            if u == target:
                break
            for v, attr in self.graph.get(u, {}).items():
                km  = attr["km"]
                hrs = attr["hrs"]

                if mode == "distance":
                    edge_cost = km

                elif mode == "time":
                    speed     = self._predict_speed(km, hrs, model_name, edge_speeds)
                    edge_cost = km / speed

                else:  # balanced — normalise both components to [0,1] before weighting
                    speed     = self._predict_speed(km, hrs, model_name, edge_speeds)
                    edge_cost = (0.6 * km / max_km) + (0.4 * (km / speed) / max_time)

                new_cost = dist[u] + edge_cost
                if new_cost < dist.get(v, INF):
                    dist[v] = new_cost
                    prev[v] = u
                    heapq.heappush(heap, (new_cost, v))

        if target not in dist:
            return None, None, None

        path, node = [], target
        while node in prev:
            path.append(node)
            node = prev[node]
        path.append(source)
        path.reverse()
        return path, dist[target], dist

    # ── public API ────────────────────────────────────────────────────────────

    def find_optimal_path(
        self,
        source:     str,
        target:     str,
        model_name: str  = None,
        commodity:  str  = None,
        rake_type:  str  = None,
    ) -> dict:
        source = source.strip().upper()
        target = target.strip().upper()
        mn     = model_name or self.best_name
        cmdt   = commodity  or DEFAULT_COMMODITY
        rkt    = rake_type  or DEFAULT_RAKE_TYPE

        if source not in self.graph:
            return {"error": f"Source '{source}' not found in route graph."}
        if target not in self._all_nodes:
            return {"error": f"Destination '{target}' not found in route graph."}

        # Resolve the correct edge-speed table and balance norms for this combo.
        edge_speeds = self._get_edge_speeds(cmdt, rkt)
        bal_norm    = (
            self._bal_norm
            if (cmdt == DEFAULT_COMMODITY and rkt == DEFAULT_RAKE_TYPE)
            else self._compute_balance_norms(edge_speeds)
        )

        results = {}
        for mode in ["distance", "time", "balanced"]:
            path, cost, _ = self._dijkstra(
                source, target,
                mode=mode, model_name=mn,
                edge_speeds=edge_speeds, bal_norm=bal_norm,
            )
            if path is None:
                results[mode] = {"path": None, "cost": None, "hops": []}
                continue

            hops, total_km, total_hrs = [], 0.0, 0.0
            for i in range(len(path) - 1):
                u, v  = path[i], path[i + 1]
                attr  = self.graph.get(u, {}).get(v, {})
                km    = attr.get("km", 0)
                hrs   = attr.get("hrs", 0)
                spd   = self._predict_speed(km, hrs, mn, edge_speeds)
                total_km  += km
                total_hrs += km / spd
                hops.append({
                    "from":      u,
                    "from_name": self.station_name(u),
                    "to":        v,
                    "to_name":   self.station_name(v),
                    "km":        round(km, 2),
                    "ml_hrs":    round(km / spd, 2),
                    "ml_speed":  round(spd, 2),
                })

            coords = self._build_coords(path)
            results[mode] = {
                "path":         path,
                "path_names":   [self.station_name(s) for s in path],
                "cost":         round(cost, 4),
                "total_km":     round(total_km, 2),
                "total_ml_hrs": round(total_hrs, 2),
                "hops":         hops,
                "coords":       coords,
                "model_used":   mn,
            }

        return {
            "source":           source,
            "source_name":      self.station_name(source),
            "target":           target,
            "target_name":      self.station_name(target),
            "commodity":        cmdt,
            "rake_type":        rkt,
            "paths":            results,
            "model_comparison": self.bundle["results"].set_index("model").to_dict("index"),
            "best_model":       self.best_name,
        }


# ── singleton loader ──────────────────────────────────────────────────────────
_finder = None


def get_finder() -> RailPathFinder:
    global _finder
    if _finder is None:
        _finder = RailPathFinder()
    return _finder


if __name__ == "__main__":
    import time
    pf = RailPathFinder()
    t0 = time.time()
    r  = pf.find_optimal_path("ABKP", "UD", commodity="STEEL", rake_type="BFNSM")
    print(f"query took {time.time() - t0:.2f}s")
    for mode, info in r["paths"].items():
        if info["path"]:
            preview = " -> ".join(info["path"][:5])
            print(
                f"[{mode.upper()}] {preview}"
                f"{'...' if len(info['path']) > 5 else ''}  "
                f"km={info['total_km']}  ml_hrs={info['total_ml_hrs']}"
            )
