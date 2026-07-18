"""Verify OpenML downloads and cache datasets."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

info = {}
for name in common.DATASETS:
    X, y = common.load_dataset(name)
    num = [c for c in X.columns
           if __import__("pandas").api.types.is_numeric_dtype(X[c])]
    info[name] = dict(n=len(X), p=X.shape[1], n_num=len(num),
                      n_cat=X.shape[1] - len(num),
                      pos_rate=float(y.mean()),
                      openml_id=common.DATASETS[name])
    print(name, info[name], flush=True)
with open(common.RES_DIR + "/dataset_info.json", "w") as f:
    json.dump(info, f, indent=2)
print("DONE")
