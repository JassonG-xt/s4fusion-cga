#!/usr/bin/env bash
# Wait for the H1 extra-seed training to finish, then run the H1 evaluation
# chain automatically (single 4 GB GPU: evaluation must not overlap training).
set -uo pipefail
cd /mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/code

echo "[$(date '+%F %T')] waiting for abl_S_s3407.done ..."
while [ ! -f checkpoints/abl_S_s3407.done ]; do
  sleep 60
done
echo "[$(date '+%F %T')] training marker present; training processes:"
pgrep -af train_brss | head -3 || true
# give the last process a moment to exit and release the GPU
sleep 20

echo "[$(date '+%F %T')] running H1 evaluation chain"
bash run_H1_eval_extra.sh
echo "[$(date '+%F %T')] H1_AFTER_ALL_DONE"
