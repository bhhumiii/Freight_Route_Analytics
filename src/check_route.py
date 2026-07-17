import pandas as pd

print("Step 1: Script started")

try:
    print("Step 2: Reading Excel...")
    df = pd.read_excel("data/OD_FY2526_VIA_LIST.xlsx")

    print("Step 3: Excel loaded successfully")
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    if "travl_route_part" in df.columns:
        print(df["travl_route_part"].head(10).tolist())
    else:
        print("'travl_route_part' column not found!")

except Exception as e:
    print("ERROR:", e)
