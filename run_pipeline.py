"""
run_pipeline.py
Master script: preprocess → train → validate
Run once before launching the Streamlit app.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("=" * 60)
print("  Indian Railways Optimal Route Finder – Pipeline")
print("=" * 60)

t0 = time.time()

# Step 1: preprocess
# Set REMOVE_OUTLIERS=0 to keep all rows; OUTLIER_METHOD=zscore to switch method.
REMOVE_OUTLIERS = os.environ.get("REMOVE_OUTLIERS", "1") != "0"
OUTLIER_METHOD  = os.environ.get("OUTLIER_METHOD", "iqr")
print("\n[1/2] Preprocessing & graph construction")
from preprocessing import run as preprocess_run
df, graph, smap = preprocess_run(outlier_method=OUTLIER_METHOD,
            cap_outliers=REMOVE_OUTLIERS)
print(f"      ✓  {len(smap):,} stations  |  {sum(len(v) for v in graph.values()):,} edges")

# Step 2: train ML models
print("\n[2/2] Training ML models (XGBoost, LightGBM, CatBoost)")
from train_models import run as train_run
bundle = train_run()
res = bundle["results"]
print("\nFinal model scores:")
print(res.to_string(index=False))

elapsed = time.time() - t0
print(f"\n{'='*60}")
print(f"  Pipeline complete in {elapsed/60:.1f} min")
print(f"  Best model: {bundle['best_model']}")
print(f"\n  Launch dashboard:")
print(f"    streamlit run src/app.py")
print("=" * 60)
