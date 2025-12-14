import pandas as pd, json
from pathlib import Path

df = pd.read_parquet("MSKArticlesINDEX/chunks.parquet")
manual = json.load(open("Eval\\article_topics_manual.json"))

stems_in_parquet = sorted({Path(p).stem.lower() if Path(p).stem.lower() != "index"
                           else Path(p).parent.name.lower() for p in df["source_relpath"].astype(str)})

manual_stems = set(manual.get("by_stem", {}))

missing = [s for s in stems_in_parquet if s not in manual_stems]

print(f"⚠️ {len(missing)} stems not found in manual map:\n")
for s in missing:
    print("-", s)
