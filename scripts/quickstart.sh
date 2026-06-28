#!/bin/bash
echo "========================================="
echo "  NFSW Detector - Quick Start"
echo "========================================="

if command -v conda &> /dev/null; then
    echo "[1/5] Conda found: $(conda --version)"
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "nfs_detector"; then
        echo "[1/5] Activating existing environment..."
        conda activate nfs_detector
    else
        echo "[1/5] Creating conda environment..."
        conda env create -f environment.yml
        conda activate nfs_detector
    fi
else
    echo "[1/5] Conda not found. Please install conda first."
    exit 1
fi

echo "[2/5] Checking CUDA/GPU..."
python -c "
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
"

echo "[3/5] Verifying key dependencies..."
python -c "
import torch, numpy, pandas, sklearn, yaml
print(f'  torch: {torch.__version__}')
print(f'  numpy: {numpy.__version__}')
print(f'  pandas: {pandas.__version__}')
print(f'  sklearn: {sklearn.__version__}')
try:
    import clip
    print('  clip: OK')
except ImportError:
    print('  clip: NOT FOUND - install with: pip install git+https://github.com/openai/CLIP.git')
try:
    import gradio
    print(f'  gradio: {gradio.__version__}')
except ImportError:
    print('  gradio: NOT FOUND')
"

echo "[4/5] Checking dataset..."
if [ -d "data/features" ] && [ "$(ls -A data/features 2>/dev/null)" ]; then
    echo "  Features directory has data"
else
    echo "  No pre-extracted features found in data/features/"
    echo "  Download from: https://huggingface.co/datasets/qiouzao/SVA"
fi

echo "[5/5] Select mode:"
echo "  1) Train model"
echo "  2) Evaluate model"
echo "  3) Detect video"
echo "  4) Launch demo"
echo "  5) Exit"
read -p "Enter choice [1-5]: " choice

case $choice in
    1) bash scripts/train.sh ;;
    2) bash scripts/evaluate.sh ;;
    3) read -p "Enter video path: " videopath; bash scripts/detect.sh "$videopath" ;;
    4) bash scripts/demo.sh ;;
    5) echo "Bye!"; exit 0 ;;
    *) echo "Invalid choice"; exit 1 ;;
esac
