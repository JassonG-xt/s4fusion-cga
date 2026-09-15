#!/bin/bash
# Wait for the B1 seed queue to finish, then run B2 distillation.
set -e
cd "$(dirname "$0")"

# M7 (2026-09-14): the wait condition below is ALREADY SATISFIED by the frozen
# checkpoints (all four exist), so this script used to fall straight through to an
# UNGUARDED `run_distill.sh` and re-run a REGISTERED FAILURE that the
# pre-registration explicitly forbids re-running. Doing so would falsify the
# manuscript's statement that this arm was "neither re-tuned nor re-run", and
# would contaminate logs/distill.log.
# Guard is the same opt-in shape as 42975ed on run_after_seeds_v2.sh.
if [ "${RUN_DISTILL:-0}" != "1" ]; then
  echo "[b2] SKIP: distillation is a registered failure (GATE1_FREEZE 搂5); not re-running it."
  echo "[b2]       Set RUN_DISTILL=1 only if you deliberately intend to re-run the failed arm."
  exit 0
fi
echo "[b2] WARNING: RUN_DISTILL=1 -- deliberately re-running the registered failure arm."

# wait for seed queue to complete (marker: all four checkpoints exist)
while [ ! -f checkpoints/abl_CGA_str_s123.pt ] || [ ! -f checkpoints/abl_CGA_str_s3407.pt ] || \
      [ ! -f checkpoints/abl_B0_s123.pt ] || [ ! -f checkpoints/abl_B0_s3407.pt ]; do
  sleep 300
done
./run_distill.sh >> logs/distill.log 2>&1
echo "[b2] distill done $(date)"
