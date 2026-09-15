#!/bin/bash
cd /mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/code
PY=.venv-brss/bin/python
export PYTHONUNBUFFERED=1
echo "=== dataset disk usage ==="
du -sh ../dataset 2>/dev/null | tail -1
echo "=== dataset dirs ==="
ls ../dataset 2>/dev/null
echo "=== WSL home free space ==="
df -h /home 2>/dev/null | tail -1
df -h /tmp 2>/dev/null | tail -1
echo "=== num images in train manifest ==="
wc -l ../dataset/manifests/train_all.csv
echo "=== quick I/O test: time to read 100 pairs via manifest (python) ==="
$PY - <<'PY'
import time, csv
from pathlib import Path
from PIL import Image
root = Path("../dataset")
with open(root/"manifests/train_all.csv", encoding="utf-8-sig") as h:
    rows = list(csv.DictReader(h))[:100]
t0 = time.time()
n = 0
for r in rows:
    Image.open(root/r["ir_path"]).convert("L").load()
    Image.open(root/r["vi_path"]).convert("RGB").load()
    n += 1
print(f"read {n} pairs in {time.time()-t0:.2f}s ({(time.time()-t0)/n*1000:.1f} ms/pair)")
PY
