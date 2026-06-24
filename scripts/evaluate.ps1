# NFSW Detector - Evaluation Script (Windows PowerShell)
# Usage: .\scripts\evaluate.ps1 [--output results/evaluation]

$ErrorActionPreference = "Stop"

# Activate conda environment
conda activate nfsw_detector 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Environment nfsw_detector not found. Run: conda env create -f environment.yml" -ForegroundColor Red
    exit 1
}

# Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Run evaluation
python main.py evaluate --config configs/default.yaml --checkpoint checkpoints/best_model.pth @args
