# NFSW Detector - Quick Start Script (Windows PowerShell)
# Usage: .\scripts\quickstart.ps1

$ErrorActionPreference = "Stop"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  NFSW Detector - Quick Start" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# [1/5] Conda environment
Write-Host "[1/5] Checking conda environment..." -ForegroundColor Green
conda activate nfsw_detector 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Creating conda environment..." -ForegroundColor Yellow
    conda env create -f environment.yml
    conda activate nfsw_detector
}
Write-Host "  Conda environment activated." -ForegroundColor Green

# [2/5] CUDA/GPU
Write-Host "[2/5] Checking CUDA/GPU..." -ForegroundColor Green
python -c @"
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
else:
    print('No GPU detected, will use CPU')
"@

# [3/5] Key dependencies
Write-Host "[3/5] Verifying key dependencies..." -ForegroundColor Green
python -c @"
import torch, numpy, pandas, sklearn, yaml
print(f'  torch: {torch.__version__}')
print(f'  numpy: {numpy.__version__}')
print(f'  pandas: {pandas.__version__}')
print(f'  sklearn: {sklearn.__version__}')
try:
    import clip
    print('  clip: OK (local module)')
except ImportError:
    print('  clip: NOT FOUND')
try:
    import gradio
    print(f'  gradio: {gradio.__version__}')
except ImportError:
    print('  gradio: NOT FOUND')
"@

# [4/5] Dataset
Write-Host "[4/5] Checking dataset..." -ForegroundColor Green
if (Test-Path "data/features") {
    $featureCount = (Get-ChildItem -Path "data/features" -Recurse -Filter "*.npy").Count
    if ($featureCount -gt 0) {
        Write-Host "  Features found: $featureCount files" -ForegroundColor Green
    } else {
        Write-Host "  No pre-extracted features found in data/features/" -ForegroundColor Yellow
        Write-Host "  Download from: https://huggingface.co/datasets/qiouzao/SVA" -ForegroundColor Yellow
    }
} else {
    Write-Host "  data/features/ directory not found" -ForegroundColor Yellow
    Write-Host "  Download from: https://huggingface.co/datasets/qiouzao/SVA" -ForegroundColor Yellow
}

# [5/5] Select mode
Write-Host "[5/5] Select mode:" -ForegroundColor Green
Write-Host "  1) Train model"
Write-Host "  2) Evaluate model"
Write-Host "  3) Detect video"
Write-Host "  4) Launch demo"
Write-Host "  5) Exit"
$choice = Read-Host "Enter choice [1-5]"

switch ($choice) {
    "1" { & "$PSScriptRoot\train.ps1" }
    "2" { & "$PSScriptRoot\evaluate.ps1" }
    "3" {
        $videoPath = Read-Host "Enter video path"
        & "$PSScriptRoot\detect.ps1" $videoPath
    }
    "4" { & "$PSScriptRoot\demo.ps1" }
    "5" { Write-Host "Bye!"; exit 0 }
    default { Write-Host "Invalid choice" -ForegroundColor Red; exit 1 }
}
