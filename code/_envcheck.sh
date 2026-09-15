#!/bin/bash
echo "=== nvidia-smi ==="
nvidia-smi 2>&1 | head -20
echo "=== python venv ==="
cd /mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/code
ls -la .venv-brss/bin/python* 2>&1
echo "=== torch ==="
.venv-brss/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')" 2>&1
echo "=== mamba_ssm ==="
.venv-brss/bin/python -c "import mamba_ssm; print('mamba_ssm ok')" 2>&1
echo "=== manifest files ==="
ls -la ../dataset/manifests/ 2>&1
echo "=== checkpoint ==="
ls -la ../../S4Fusion-main/model/model.pkl 2>&1
