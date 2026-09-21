#!/usr/bin/env bash
# Start the H1 extra-seed run fully detached from the wsl.exe session, so a
# host-side session teardown cannot SIGHUP it. Marker-driven script: safe to
# re-invoke, and it resumes from <save-path>_last.
cd /mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/code
mkdir -p logs
setsid nohup bash run_H1_seeds_extra.sh > logs/H1_extra_detached.log 2>&1 < /dev/null &
sleep 3
echo "detached pid(s):"
pgrep -af "run_H1_seeds_extra|train_brss" | head -5
