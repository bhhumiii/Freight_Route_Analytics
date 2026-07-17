# 🚂 Indian Railways Optimal Freight Route Finder
### CRIS Internship Project — Smart Rake Routing with ML

---

## Overview
This project finds the **optimal freight route** between any two Indian Railways stations
using a graph built from 4.4 lakh real rake movement records.  
Three ML models predict **circuit speed** on each edge:

| Model | Role |
|-------|------|
| **XGBoost** | Best performer (lowest MAE) |
| **LightGBM** | Fast, competitive accuracy |
| **CatBoost** | Native categorical handling |

The user inputs source & destination station codes; the system returns:
- **Shortest distance** path (Dijkstra on km)
- **Fastest time** path (Dijkstra on ML-predicted hours)
- **Balanced** path (0.5·distance + 0.5·time composite)

---

## Project Structure
```
railroute/
├── data/
│   ├── railrake.csv          ← raw input (441k rows, 81 cols; any *.csv here is auto-detected)
│   ├── processed.pkl         ← cleaned dataframe (auto-generated)
│   ├── graph.pkl             ← station adjacency graph (auto-generated)
│   └── station_map.pkl       ← code → full name map (auto-generated)
├── models/
│   └── models_bundle.pkl     ← all 3 trained models (auto-generated)
├── src/
│   ├── preprocessing.py      ← data cleaning + graph builder
│   ├── train_models.py       ← XGBoost / LightGBM / CatBoost training
│   ├── pathfinder.py         ← Dijkstra path-finding engine
│   └── app.py                ← Streamlit dashboard
├── run_pipeline.py           ← master runner (preprocess + train)
└── requirements.txt
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the pipeline (once)
```bash
python run_pipeline.py
```
This preprocesses the CSV, builds the route graph, and trains all 3 models (~1 min).

### 3. Launch the dashboard
```bash
streamlit run src/app.py
```
Open http://localhost:8501 in your browser.

---

## Model Results (40k validation rows, after IQR outlier winsorising)

| Model    | MAE   | MSE   | RMSE  | RMSLE | R²    |
|----------|-------|-------|-------|-------|-------|
| XGBoost  | 1.386 | 3.834 | 1.958 | 0.236 | 0.826 |
| LightGBM | 1.421 | 3.975 | 1.994 | 0.241 | 0.820 |
| CatBoost | 1.446 | 4.081 | 2.020 | 0.245 | 0.815 |

**Target**: `circuit_speed` (km/h) — proxy for route efficiency.
R² reflects a model trained on IQR-winsorised columns: ~2–6% of extreme values
per column are capped to the Tukey fences rather than dropped, so the graph keeps
full connectivity while the ML target is de-noised toward typical routes.

---

## Key Features
- **3,555 unique stations** across all Indian Railway zones
- **35,636 directed edges** with real median km, hours, speed
- ML models select optimal model automatically (best MAE)
- User can switch between XGBoost / LightGBM / CatBoost live
- Hop-by-hop breakdown with ML-predicted speed per segment
- Station search by code or name

---

## Data Columns Used
| Column | Use |
|--------|-----|
| `ldngsttn` / `uldg_sttn` | Source / destination station codes |
| `ldng_uldg_km` | Direct route distance (km) |
| `ldng_uldg_hor` | Travel hours |
| `circuit_speed` | **ML target** — km/h efficiency |
| `circuittime` | Full circuit time (hh:mm) |
| `circuitkm` | Full circuit distance |
| `ldngzone` / `uldgzone` | Railway zone (feature) |
| `raketype` / `grupcmdt` | Wagon type / commodity (feature) |

---

*Built by bhumi — CRIS Internship, Freight Domain*
