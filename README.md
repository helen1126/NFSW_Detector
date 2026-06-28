# NFSW Detector

**面向短视频社交平台的多模态有害内容审查与预警系统**

基于 SVLA（Shot-Conditioned Vision-Language Adaptation）模型和 CLIP 视觉-语言预训练模型，对短视频中的有害内容进行自动识别，输出异常分数与有害时间段定位，生成分级预警报告，并提供基于 Gradio 的交互式 Web Demo。

## 有害内容类别

| ID | 英文 | 中文 | 描述 |
|----|------|------|------|
| 0 | Smoke | 吸烟 | 疑似吸烟行为，违反平台健康规范 |
| 1 | Blood | 血腥 | 血腥画面，可能引起观众不适 |
| 2 | Violent | 暴力 | 暴力行为，违反平台社区准则 |
| 3 | Abusive | 辱骂 | 辱骂行为，可能构成言语骚扰 |
| 4 | Sexy | 色情 | 色情内容，违反平台内容政策 |
| 5 | Money | 金钱诈骗 | 疑似金钱诈骗内容，存在欺诈风险 |
| 6 | Policy | 政治敏感 | 政治敏感内容，可能违反相关规定 |

## 项目结构

```
NFSW_Detector/
├── clip/                      CLIP 模型（本地模块，含 encode_token）
│   ├── clip.py
│   ├── model.py
│   ├── simple_tokenizer.py
│   └── bpe_simple_vocab_16e6.txt.gz
├── configs/                  配置文件
│   └── default.yaml          默认配置
├── data/                     数据目录
│   ├── raw/                  原始视频
│   ├── features/             CLIP 预提取特征 (.npy)
│   ├── splits/               训练/测试划分 CSV
│   └── dataset.py            SVA 数据集加载器
├── engine/                   训练/测试引擎
│   ├── train.py              训练逻辑 (MIL 损失, 梯度裁剪, 早停)
│   └── evaluate.py           评估逻辑 (AUC, AP, 可视化)
├── list/                     GT 文件（帧级标签）
│   ├── gt_Sva_500_abnormal_is_1.npy
│   └── gt_Sva_small.npy
├── models/                   模型定义
│   ├── svla.py               SVLA 核心模型
│   ├── layers.py             图卷积层 (GraphConvolution, DistanceAdj)
│   └── classifier.py         多标签分类头与损失函数
├── pipeline/                 推理管线
│   ├── preprocess.py         视频预处理与帧采样
│   ├── feature_extractor.py  CLIP 特征提取
│   ├── inference.py          端到端推理 (NSFWDetector)
│   ├── alert.py              预警信息生成
│   └── calibration.py        分数校准 (ScoreCalibrator)
├── demo/                     Demo 系统
│   ├── app.py                Gradio 主入口
│   └── visualize.py          Plotly 可视化工具
├── api/                      FastAPI RESTful API 服务
│   ├── app.py                FastAPI 应用与路由
│   └── schemas.py            Pydantic 请求/响应模型
├── utils/                    工具函数
│   ├── tools.py              SVLA 工具 (get_prompt_text, process_feat 等)
│   ├── video.py              视频解码/格式转换
│   ├── metrics.py            AUC/AP 计算
│   └── logger.py             日志工具
├── tests/                    单元测试
├── scripts/                  运行脚本
│   ├── train.sh / train.ps1
│   ├── evaluate.sh / evaluate.ps1
│   ├── demo.sh / demo.ps1
│   ├── detect.sh / detect.ps1
│   ├── quickstart.sh / quickstart.ps1
│   ├── generate_splits.py           随机划分生成
│   ├── generate_reference_splits.py 参考划分生成（推荐）
│   └── fit_calibrator.py            离线拟合分数校准器
├── docs/                     项目文档
│   ├── API.md                前端对接 API 文档
│   └── ARCHITECTURE.md       模块结构与数据流
├── main.py                   CLI 入口
├── setup.py                  安装配置
├── environment.yml           Conda 环境配置
└── requirements.txt          Pip 依赖列表
```

## 环境要求

- Python >= 3.9
- PyTorch >= 2.0
- CUDA >= 11.8（推荐 12.1）

## 安装

### 方式一：Conda（推荐）

```bash
conda env create -f environment.yml
conda activate nfsw_detector
```

**Windows 用户注意：** CLIP 包需要手动安装，详见 [INSTALL.md](INSTALL.md)。

### 方式二：Pip

```bash
pip install -r requirements.txt
```

### 快速启动

**Linux/macOS:**
```bash
bash scripts/quickstart.sh
```

**Windows (PowerShell):**
```powershell
.\scripts\quickstart.ps1
```

该脚本将自动检查 conda 环境、CUDA/GPU、关键依赖和数据集，并提供交互式模式选择。

**Windows 手动命令：**

如果不想使用脚本，可手动执行以下命令：

```powershell
# 1. 创建并激活 Conda 环境
conda env create -f environment.yml
conda activate nfsw_detector

# 2. 检查 CUDA/GPU
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 3. 验证依赖
python -c "import torch, numpy, pandas, sklearn, yaml; print('Core dependencies OK')"

# 4. 检查数据集
ls data\features

# 5. 选择运行模式
# 训练
python main.py train --config configs/default.yaml

# 评估
python main.py evaluate --config configs/default.yaml --checkpoint checkpoints/best_model.pth

# 单视频检测
python main.py detect --config configs/default.yaml --checkpoint checkpoints/best_model.pth --video test.mp4

# 启动 Demo
python main.py demo --config configs/default.yaml --checkpoint checkpoints/best_model.pth
```

**注意：** CLI 命令中的方括号 `[...]` 表示可选参数，实际运行时请勿包含方括号。例如：
```powershell
# 错误写法
python main.py train --config configs/default.yaml [--device cuda:0]

# 正确写法
python main.py train --config configs/default.yaml --device cuda:0
```

## 数据准备

从 HuggingFace 下载 SVA 数据集的预提取 CLIP 特征：

```bash
# 下载后放置到 data/features/ 目录
# 下载地址: https://huggingface.co/datasets/qiouzao/SVA
```

### 生成训练/测试划分

项目使用参考实现的固定划分（502 个测试视频，与 GT 文件对齐）：

```bash
# 生成参考划分（推荐，与 GT 文件对齐）
python scripts/generate_reference_splits.py

# 或生成随机划分（不推荐，需自行构建 GT）
python scripts/generate_splits.py
```

CSV 格式（`data/splits/train.csv` 和 `data/splits/test.csv`）：

```csv
path,label
data/features/clip_features/smoke/video001.npy,smoke
data/features/clip_features/blood/video002.npy,blood
data/features/clip_features/normal/video003.npy,normal
...
```

### GT 文件

帧级 GT 文件位于 `list/` 目录，用于评估时计算 AUC/AP：
- `list/gt_Sva_500_abnormal_is_1.npy`：502 个测试视频的帧级二分类标签（0=normal, 1=abnormal）
- 路径在 `configs/default.yaml` 的 `data.gt_path` 中配置

## 使用方法

### CLI 命令

```bash
# 训练
python main.py train --config configs/default.yaml [--resume checkpoint.pth] [--device cuda:0]

# 评估
python main.py evaluate --config configs/default.yaml --checkpoint checkpoints/best_model.pth [--output results/evaluation]

# 单视频检测
python main.py detect --config configs/default.yaml --checkpoint checkpoints/best_model.pth --video test.mp4 [--threshold 0.5] [--output results/detection]

# 启动 Gradio Demo
python main.py demo --config configs/default.yaml [--checkpoint checkpoints/best_model.pth] [--port 7860] [--share]

# 启动 RESTful API 服务（Swagger 文档: http://localhost:8000/docs）
python main.py serve --config configs/default.yaml [--checkpoint checkpoints/best_model.pth] [--host 0.0.0.0] [--port 8000]

# 模型导出
python main.py export --checkpoint checkpoints/best_model.pth --format onnx [--output exports/model]
```

### 骨干网络选择

支持两种 CLIP 骨干网络，通过配置文件切换：

| 配置文件 | 骨干 | 维度 | 说明 |
|----------|------|------|------|
| `configs/default.yaml` | ViT-B/16 | 512 | 默认，轻量快速 |
| `configs/vitl14.yaml` | ViT-L/14 | 768 | 高精度，参数量约 2 倍 |

```bash
# 使用高精度骨干训练（需重新训练，检查点不兼容）
python main.py train --config configs/vitl14.yaml
```

**注意：** 切换骨干网络后需重新训练，ViT-B/16 的检查点无法直接用于 ViT-L/14 配置。配置校验会在模型构造前检查 `clip_variant` 与 `embed_dim`/`visual_width` 的一致性。

### Shell 脚本

**Linux/macOS:**
```bash
bash scripts/train.sh
bash scripts/evaluate.sh
bash scripts/detect.sh <video_path>
bash scripts/demo.sh
```

**Windows (PowerShell):**
```powershell
.\scripts\train.ps1
.\scripts\evaluate.ps1
.\scripts\detect.ps1 <video_path>
.\scripts\demo.ps1
```

### Python API

```python
import yaml
from pipeline.inference import NSFWDetector
from pipeline.alert import AlertGenerator

with open("configs/default.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

detector = NSFWDetector(config, checkpoint_path="checkpoints/best_model.pth")
result = detector.detect("test_video.mp4")

alert_gen = AlertGenerator(config)
report = alert_gen.generate(result)
alert_gen.export_json(report, "report.json")

print(f"异常分数: {result.anomaly_score:.4f}")
print(f"是否有害: {result.is_harmful}")
print(f"预测类别: {result.predicted_categories}")
print(f"类别分数: {result.category_scores}")
print(f"有害时间段: {len(result.harmful_segments)} 段")
```

> 完整字段定义、JSON 报告结构与前端对接说明见 [docs/API.md](docs/API.md)。

## 模型架构

### SVLA (Shot-Conditioned Vision-Language Adaptation)

```
输入视频 → CLIP 特征提取 → Temporal Transformer → 镜头检测 → Shot Transformer
                                                                    ↓
                                              ┌─────────────────────┤
                                              ↓                     ↓
                                     Shot Density Head      Shot-Aware GCN
                                     (异常密度估计)          (图卷积网络)
                                              ↓                     ↓
                                        MIL Aggregator        CFATextAdapter
                                        (多实例学习)          (视觉-语言适配)
                                              ↓                     ↓
                                        异常分类头             类别分类头
                                              ↓                     ↓
                                        视频级异常分数        7类有害内容预测
                                        段级异常曲线          类别概率分布
```

核心组件：

- **ShotConditionModule**：基于余弦相似度的镜头边界检测 + ShotDensityHead 密度估计 + Shot Transformer 段内增强
- **VisionLanguageAdapter (CFATextAdapter)**：交叉注意力机制（Q=视觉特征，K/V=文本特征+动态 Prefix），门控融合
- **TextCFAdapter**：视觉条件文本特征适配（FiLM 风格）
- **MILAggregator**：图卷积 + 距离邻接矩阵 + 门控注意力聚合

### 损失函数

- **CLAS2**：二分类异常检测损失（Shot-aware MIL Pooling + BCE）
- **CLASM**：多类别分类损失（Shot-aware MIL Pooling + NLL）
- **Text Regularizer**：文本特征正则化（最小化 normal 与其他类别特征的余弦相似度）

总损失 = CLAS2 + α × CLASM + Text Regularizer

## 预警等级

| 等级 | 分数范围 | 颜色 | 处置建议 |
|------|----------|------|----------|
| HIGH | ≥ 0.8 | 红色 | 立即下架并转人工审核 |
| MEDIUM | ≥ 0.5 | 橙色 | 限制推荐并人工复审 |
| LOW | ≥ 0.3 | 黄色 | 标记待审，纳入审核队列 |
| SAFE | < 0.3 | 绿色 | 常规监控 |

## Gradio Demo

```bash
python main.py demo --checkpoint checkpoints/best_model.pth
```

访问 `http://localhost:7860`，功能包括：

- **检测概览**：预警等级标签、异常分数、有害类别列表、处置建议
- **时间轴分析**：Plotly 交互异常分数曲线、视频播放器
- **关键帧**：Gallery 展示有害时间段关键帧
- **详细报告**：完整预警报告、JSON/HTML 下载

## 配置说明

所有配置在 `configs/default.yaml` 中，主要配置段：

| 配置段 | 说明 | 关键参数 |
|--------|------|----------|
| model | 模型架构 | clip_variant, embed_dim, visual_length, shot_sim_thresh |
| data | 数据路径 | feature_dir, train_csv, test_csv, gt_path, supported_formats |
| training | 训练参数 | batch_size, lr, epochs, warmup_epochs, class_loss_alpha |
| cuda | GPU 配置 | device, amp, tf32, cudnn_benchmark |
| inference | 推理参数 | anomaly_threshold, alert_levels, max_duration |
| demo | Demo 配置 | port, share, max_file_size |
| logging | 日志配置 | level, log_file, tensorboard |
| calibration | 分数校准 | enabled, path |
| ood | OOD 检测 | enabled, threshold |
| frame_quality | 帧质量加权 | enabled |
| zero_shot | 零样本扩展 | enabled, extra_categories |

## 推理增强（无须重训）

支持 4 个推理时增强功能，通过配置开关启用，默认全部关闭以保证向后兼容：

| 增强项 | 配置段 | 说明 |
|--------|--------|------|
| 分数校准 | `calibration` | Isotonic Regression 把 raw sigmoid 映射到真实概率 |
| OOD 检测 | `ood` | 基于类别分布熵识别分布外内容 |
| 关键帧质量加权 | `frame_quality` | 模糊/过暗帧降权 |
| 零样本新类别 | `zero_shot` | CLIP 文本-图像相似度检测训练集外类别 |

### 分数校准

校准器把模型的 raw sigmoid 分数映射到真实概率，提升分数的可解释性。需先离线拟合：

```bash
python scripts/fit_calibrator.py --config configs/default.yaml \
    --checkpoint checkpoints/best_model.pth --output checkpoints/calibrator.pkl
```

然后在配置中启用：
```yaml
calibration:
  enabled: true
  path: checkpoints/calibrator.pkl
```

### OOD 检测

基于类别分布的熵 + 最大概率识别分布外内容（如训练集未见过的场景）。`ood_score >= threshold` 时标记为 OOD。

### 关键帧质量加权

综合 Laplacian 方差（清晰度）、亮度合理性、Shannon 熵评估每帧质量，对低质量帧的异常分数降权。

### 零样本新类别扩展

通过 CLIP 文本-图像相似度检测训练集外的自定义类别（如赌博、毒品），无须重新训练。在 `zero_shot.extra_categories` 中配置类别名、文本提示和中文标签。

### 采样帧数可选

CLI 和 API 支持请求级覆盖采样帧数：

```bash
# CLI
python main.py detect --config configs/default.yaml --checkpoint checkpoints/best_model.pth \
    --video test.mp4 --num-segments 20

# API
curl -X POST http://localhost:8000/api/v1/detect \
    -F "file=@test.mp4" -F "num_segments=20"
```

### 图片检测 API

支持对单张图片进行有害内容检测（视为单帧视频复用推理管线）：

```bash
curl -X POST http://localhost:8000/api/v1/detect_image \
    -F "file=@test.jpg"
```

支持的图片格式：jpg, jpeg, png, bmp, webp。返回结构与视频检测一致。

## 测试

```bash
pytest tests/ -v
pytest tests/ -v -m "not slow"     # 跳过耗时测试
pytest tests/ -v -m "not gpu"      # 跳过需 GPU 测试
```

## 性能指标目标

| 指标 | 目标 |
|------|------|
| 视频级异常检测 AUC | ≥ 85% |
| Average Precision | ≥ 80% |
| 推理速度 | ≤ 5秒/分钟视频（单 GPU） |
| 支持视频格式 | ≥ 5 种 (MP4/AVI/MOV/FLV/MKV) |
| 识别有害类别 | ≥ 4 类 (实际支持 7 类) |

## 参考文献

- **SVLA**: Shot-Conditioned Vision-Language Adaptation for Anomaly Detection in User-Generated Content (IJCAI)
- **CLIP**: Learning Transferable Visual Models From Natural Language Supervision (ICML 2021)
- **DeepMIL**: Attention-based Deep Multiple Instance Learning (Ilse et al., 2018)

## License

MIT
