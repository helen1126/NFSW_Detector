# 安装指南

## Windows 环境安装步骤

### 1. 创建 Conda 环境

`environment.yml` 中默认环境名为 `nfs_detector`，可按需修改。下面统一使用 `nfsw_detector` 作为示例：

```powershell
conda env create -f environment.yml
conda activate nfsw_detector
```

### 2. CLIP 模块

项目自带本地 CLIP 模块（`clip/` 目录），包含 SVLA 所需的 `encode_token` 和双参数 `encode_text` 方法，**无需额外安装 OpenAI CLIP**。

如需验证：
```powershell
python -c "import clip; print('CLIP 本地模块 OK')"
```

**注意：** CLIP tokenizer 需要 `clip/bpe_simple_vocab_16e6.txt.gz` 文件。项目已包含此文件（实际是 gzip 压缩的词汇表）。如缺失，可从 `bpe_simple_vocab_16e6.txt` 复制：
```powershell
Copy-Item "clip\bpe_simple_vocab_16e6.txt" "clip\bpe_simple_vocab_16e6.txt.gz"
```

### 3. 验证安装

```powershell
python -c "import torch, clip; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print('CLIP: OK')"
```

### 4. 生成数据划分

```powershell
# 生成参考划分（推荐，与 GT 文件对齐）
python scripts/generate_reference_splits.py

# 验证划分
python -c "import pandas as pd; df=pd.read_csv('data/splits/test.csv'); print(f'Test: {len(df)} videos')"
```

预期输出：`Test: 502 videos`

## Windows PowerShell 脚本

项目提供 5 个 PowerShell 脚本，对应 Linux/macOS 的 bash 脚本：

| 脚本 | 用途 | 用法 |
|------|------|------|
| `scripts/quickstart.ps1` | 快速启动引导 | `.\scripts\quickstart.ps1` |
| `scripts/train.ps1` | 训练模型 | `.\scripts\train.ps1` |
| `scripts/evaluate.ps1` | 评估模型 | `.\scripts\evaluate.ps1` |
| `scripts/detect.ps1` | 单视频检测 | `.\scripts\detect.ps1 <video_path>` |
| `scripts/demo.ps1` | 启动 Gradio Demo | `.\scripts\demo.ps1` |

**使用示例：**
```powershell
# 快速启动（交互式选择）
.\scripts\quickstart.ps1

# 直接训练
.\scripts\train.ps1

# 评估
.\scripts\evaluate.ps1

# 检测视频
.\scripts\detect.ps1 test.mp4
```

**注意：** 如遇到 PowerShell 执行策略限制，运行：
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## 常见问题

### Q: CUDA 不可用

**A:** 确保：
1. 安装了 NVIDIA 驱动
2. PyTorch CUDA 版本与系统 CUDA 版本匹配
3. 检查 CUDA：
```powershell
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Q: decord 安装失败

**A:** Windows 上 decord 可能需要 Visual C++ Build Tools。替代方案：
```powershell
pip install eva-decord
```

### Q: 网络问题导致下载慢

**A:** 使用国内镜像：
```powershell
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### Q: 训练时 GPU 显存不足 (OOM)

**A:** 降低 `configs/default.yaml` 中的 `batch_size`：
```yaml
training:
  batch_size: 16  # 从 32 降至 16
```

### Q: 评估时 AUC 为 0.0

**A:** 检查以下几点：
1. 确认 `list/gt_Sva_500_abnormal_is_1.npy` 文件存在
2. 确认 `data/splits/test.csv` 使用参考划分（502 个视频）：
   ```powershell
   python scripts/generate_reference_splits.py
   ```
3. 确认 `configs/default.yaml` 中 `data.gt_path` 配置正确
4. 确认 GT 文件路径与测试集划分对齐（路径错误会导致全部样本被忽略）

### Q: 训练时报 "AMP 与 F.binary_cross_entropy 不兼容"

**A:** SVLA 模型在 AMP 模式下与 `F.binary_cross_entropy` 冲突。请确认 `configs/default.yaml` 中 `cuda.amp: false`（默认已禁用），切勿开启。

### Q: DataLoader 报错 "error code 1455" 或共享内存不足

**A:** Windows 共享内存有限。将 `configs/default.yaml` 中 `cuda.num_workers` 调小为 2，并保持 `data/dataset.py` 中 `persistent_workers=True`、`prefetch_factor=2`（仅当 `num_workers>0` 时设置），可显著降低共享内存占用。

### Q: 训练时报 "in-place operation 导致梯度计算错误"

**A:** SVLA 模型 `models/svla.py` 的 `_mask_row_normalize` 等函数禁止使用 in-place 修改（如 `x /= ...`、`x[...] = ...`）。若自行修改模型代码，请改用 out-of-place 写法（如 `x = x / norm`）。

### Q: 训练时 loss 不下降 / 类别分数全部相同

**A:** 检查 `configs/default.yaml` 中 `label_map` 顺序是否与训练/推理一致（`get_prompt_text` 返回 `label_map.values()` 列表）。推理时 `text_list` 必须与训练时一致，否则会导致类别索引错位。

## 完整安装流程（Windows）

```powershell
# 1. 创建环境
conda env create -f environment.yml
conda activate nfsw_detector

# 2. 验证安装
python -c "import torch, clip; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print('CLIP: OK')"

# 3. 生成数据划分
python scripts/generate_reference_splits.py

# 4. 检查数据
ls data\features

# 5. 运行训练
python main.py train --config configs/default.yaml
```
