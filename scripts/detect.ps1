# NFSW Detector - Single Video Detection Script (Windows PowerShell)
# Usage: .\scripts\detect.ps1 <video_path> [--threshold 0.5] [--output results/detection]

$ErrorActionPreference = "Stop"

if ($args.Count -eq 0) {
    Write-Host "Usage: .\scripts\detect.ps1 <video_path> [--threshold <float>] [--output <dir>]" -ForegroundColor Yellow
    exit 1
}

$videoPath = $args[0]
$remainingArgs = $args[1..($args.Count - 1)]

# Activate conda environment
conda activate nfsw_detector 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Environment nfsw_detector not found. Run: conda env create -f environment.yml" -ForegroundColor Red
    exit 1
}

# Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Run detection
python main.py detect --config configs/default.yaml --checkpoint checkpoints/best_model.pth --video $videoPath @remainingArgs
