#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-../dataset}"
DOWNLOADS="$ROOT/_downloads"
GDOWN="${GDOWN:-../.dataset-tools-venv/bin/gdown}"
mkdir -p "$DOWNLOADS" "$ROOT/LLVIP" "$ROOT/M3FD" "$ROOT/FMB" "$ROOT/MSRS"

free_gb() { df -BG "$ROOT" | awk 'NR==2 {gsub(/G/,"",$4); print $4}'; }
require_space() {
  local available
  available="$(free_gb)"
  if [ "$available" -lt 12 ]; then
    echo "Refusing to continue: only ${available}GB free; 12GB safety reserve required." >&2
    exit 2
  fi
}

require_space
wget -c -O "$DOWNLOADS/LLVIP.zip" \
  "https://huggingface.co/datasets/jsonhash/LLVIP/resolve/main/LLVIP.zip?download=true"
echo "b3b55475c093ac54663c467859eb31a82b840b53a7196ad1887cb547e4518b45  $DOWNLOADS/LLVIP.zip" | sha256sum -c -
unzip -q "$DOWNLOADS/LLVIP.zip" -d "$ROOT/LLVIP"
rm "$DOWNLOADS/LLVIP.zip"

require_space
"$GDOWN" -c "https://drive.google.com/uc?id=1pjdhjVTpOsj2qMBVIRpLOLA7UWIuHt0P" -O "$DOWNLOADS/M3FD_Fusion.zip"
sha256sum "$DOWNLOADS/M3FD_Fusion.zip" > "$ROOT/audits/M3FD_Fusion.sha256"
unzip -tq "$DOWNLOADS/M3FD_Fusion.zip"
unzip -q "$DOWNLOADS/M3FD_Fusion.zip" -d "$ROOT/M3FD"
rm "$DOWNLOADS/M3FD_Fusion.zip"

require_space
"$GDOWN" -c "https://drive.google.com/uc?id=1C8kkYkj1Xls6UtvJ4h6UajiPcvaQ7eeI" -O "$DOWNLOADS/M3FD_Detection.zip"
sha256sum "$DOWNLOADS/M3FD_Detection.zip" > "$ROOT/audits/M3FD_Detection.sha256"
unzip -tq "$DOWNLOADS/M3FD_Detection.zip"
mkdir -p "$ROOT/M3FD/full"
unzip -q "$DOWNLOADS/M3FD_Detection.zip" -d "$ROOT/M3FD/full"
rm "$DOWNLOADS/M3FD_Detection.zip"

require_space
"$GDOWN" -c "https://drive.google.com/uc?id=1nk2R2yZ05SDe2lDpsk2D5RAngMBPUPsB" -O "$DOWNLOADS/FMB_train.zip"
"$GDOWN" -c "https://drive.google.com/uc?id=1AamLVgMG5JCkhUOZi-xKTymZXCNklnaq" -O "$DOWNLOADS/FMB_test.zip"
sha256sum "$DOWNLOADS/FMB_train.zip" "$DOWNLOADS/FMB_test.zip" > "$ROOT/audits/FMB.sha256"
unzip -tq "$DOWNLOADS/FMB_train.zip"
unzip -tq "$DOWNLOADS/FMB_test.zip"
unzip -q "$DOWNLOADS/FMB_train.zip" -d "$ROOT/FMB/train"
unzip -q "$DOWNLOADS/FMB_test.zip" -d "$ROOT/FMB/test"
rm "$DOWNLOADS/FMB_train.zip" "$DOWNLOADS/FMB_test.zip"

require_space
if [ ! -d "$ROOT/MSRS/.git" ]; then
  git clone --depth 1 https://github.com/Linfeng-Tang/MSRS.git "$ROOT/MSRS"
fi

python3 "$(dirname "$0")/prepare_datasets.py" --data-root "$ROOT"
