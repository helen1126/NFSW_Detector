#!/bin/bash
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate nfs_detector 2>/dev/null || { echo "Environment nfs_detector not found. Run: conda env create -f environment.yml"; exit 1; }
else
    echo "Conda not found. Please install conda first."
    exit 1
fi

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

python main.py demo --config configs/default.yaml --checkpoint checkpoints/best_model.pth "$@"
