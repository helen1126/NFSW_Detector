# NFSW Detector 模块架构

本文档说明项目的模块划分、依赖关系与数据流，便于二次开发与代码定位。

- 关联文档：[README.md](../README.md) · [API.md](API.md)

---

## 1. 顶层模块依赖

```
            ┌─────────────┐
            │   main.py   │  CLI 入口（argparse 子命令分发）
            └──────┬──────┘
       ┌────────────┼────────────┐
       ▼            ▼            ▼
  engine/train   engine/     pipeline/
                 evaluate    inference
       │            │            │
       │            │       ┌────┴─────┬────────────┐
       │            │       ▼          ▼            ▼
       │            │  pipeline/   pipeline/    pipeline/
       │            │  preprocess  feature_     alert
       │            │              extractor
       │            │            │
       ▼            ▼            ▼
  data/dataset   models/svla ── models/classifier
       │            │                ▲
       │            │                │
       └────────────┴────────────────┘
                    │
              utils/tools (process_feat, get_prompt_text)
              utils/metrics (AUC/AP)
              utils/logger
              clip/ (本地 CLIP 模块)
```

依赖方向自上而下；`utils/`、`clip/` 为底层共享模块。

---

## 2. 模块职责

### 2.1 入口与编排

| 文件 | 职责 |
|------|------|
| [main.py](../main.py) | CLI 子命令 `train / evaluate / detect / demo / export`，组装 config、模型、数据加载器 |
| [configs/default.yaml](../configs/default.yaml) | 模型/训练/推理/Demo 全部参数 |

### 2.2 数据层

| 文件 | 职责 |
|------|------|
| [data/dataset.py](../data/dataset.py) | SVA 数据集加载，`create_dataloaders()` 返回 normal/anomaly/test 三个 loader |
| [scripts/generate_reference_splits.py](../scripts/generate_reference_splits.py) | 生成参考划分（与 GT 对齐，推荐） |
| [scripts/generate_splits.py](../scripts/generate_splits.py) | 生成随机划分 |
| [list/gt_Sva_500_abnormal_is_1.npy](../list/gt_Sva_500_abnormal_is_1.npy) | 502 测试视频的帧级二分类标签 |

### 2.3 模型层

| 文件 | 职责 |
|------|------|
| [models/svla.py](../models/svla.py) | SVLA 主模型：ShotConditionModule + VisionLanguageAdapter + MILAggregator + 双分类头 |
| [models/classifier.py](../models/classifier.py) | `LABEL_MAP` 常量、`MultiLabelClassifier`（备用分类头）、`MultiLabelLoss` |
| [models/layers.py](../models/layers.py) | 图卷积层 `GraphConvolution`、`DistanceAdj` |
| [clip/](../clip/) | 本地 CLIP 模块（含 `encode_token` 与双参数 `encode_text`，**不要安装 OpenAI CLIP**） |

### 2.4 训练/评估引擎

| 文件 | 职责 |
|------|------|
| [engine/train.py](../engine/train.py) | `Trainer`：MIL 损失（CLAS2 + α·CLASM + 文本正则）、MultiStepLR、早停、checkpoint |
| [engine/evaluate.py](../engine/evaluate.py) | `Evaluator`：AUC/AP 计算、结果导出 |

### 2.5 推理管线

| 文件 | 职责 |
|------|------|
| [pipeline/preprocess.py](../pipeline/preprocess.py) | `VideoPreprocessor`：解码、采样、抽帧、长视频降采样 |
| [pipeline/feature_extractor.py](../pipeline/feature_extractor.py) | `CLIPFeatureExtractor`：批量视觉特征提取 |
| [pipeline/inference.py](../pipeline/inference.py) | `NSFWDetector`：端到端推理，输出 `DetectionResult` |
| [pipeline/alert.py](../pipeline/alert.py) | `AlertGenerator`：规则化整理为 `AlertReport`，导出 JSON/HTML/CSV |

### 2.6 Demo

| 文件 | 职责 |
|------|------|
| [demo/app.py](../demo/app.py) | Gradio App 主入口，单例持有 `NSFWDetector` |
| [demo/visualize.py](../demo/visualize.py) | Plotly 可视化（异常曲线、类别分布、注意力热图、仪表盘、报告 HTML） |

### 2.7 工具层

| 文件 | 职责 |
|------|------|
| [utils/tools.py](../utils/tools.py) | `get_prompt_text`（取 `label_map.values()`）、`process_feat`（uniform_extract + pad 到 `visual_length`）、`get_batch_mask` |
| [utils/metrics.py](../utils/metrics.py) | AUC / AP 计算 |
| [utils/logger.py](../utils/logger.py) | 日志配置 |
| [utils/video.py](../utils/video.py) | 视频解码/格式转换辅助 |

---

## 3. 推理数据流

```
video_path (str)
    │
    ▼ NSFWDetector.detect()
VideoPreprocessor.preprocess()
    │  → frames: List[np.ndarray]  (T, H, W, 3)
    │  → timestamps: List[(start, end)]
    │  → fps, duration
    ▼
CLIPFeatureExtractor.extract_visual_features(batch)
    │  → features: np.ndarray (T, embed_dim=512)
    ▼
process_feat(features, visual_length=256)
    │  → features: (256, 512), valid_length: int
    ▼
SVLA(features_tensor, padding_mask, text_list, lengths)
    │  → logits1: (B, T, 1)    二分类异常 logit
    │  → logits2: (B, T, 8)    8 类（含 normal）logit
    │  → attn_weights
    ▼
segment_scores = sigmoid(logits1[:, 0])        段级异常分数
class_scores_raw = softmax(logits2)[0]         段级类别分布
anomaly_score = max(segment_scores)            视频级异常分数
    │
    ▼ 取异常分数最高帧的类别分布
category_scores[cat] = anomaly_score × P(cat|anomaly)
    │  其中 P(cat|anomaly) = softmax[i] / (1 - softmax[normal])
    ▼
_locate_harmful_segments()    基于 threshold + 滑动窗 + 区域合并
    │  → harmful_segments: List[HarmfulSegment]
    ▼
_extract_keyframes()          cv2 抽取每段峰值帧保存为 jpg
    │  → keyframe_paths: List[str]
    ▼
DetectionResult (dataclass)
    │
    ▼ AlertGenerator.generate()
_normalize_input()  asdict 转换 + 字段适配
    │  → categories, time_segments, keyframe_path
_determine_level()  SAFE/LOW/MEDIUM/HIGH
_extract_harmful_contents()  逐类别组装 HarmfulContentDetail
_generate_summary()  中文摘要
    ▼
AlertReport (dataclass)
    │
    ▼ export_json / export_html / export_csv
持久化文件
```

---

## 4. 训练数据流

```
data/splits/{train,test}.csv
    │
    ▼ create_dataloaders(config)
SVA Dataset  →  normal_loader / anomaly_loader / test_loader
    │  (DataLoader 配置: num_workers=2, persistent_workers=True,
    │   prefetch_factor=2 — Windows 共享内存约束)
    ▼
每 batch: (features, texts, lengths)
    │
    ▼ SVLA(features, padding_mask, text_list, lengths)
logits1, logits2, text_features
    │
    ▼
CLAS2 Loss  = BCE(sigmoid(MIL_pool(logits1)), abnormal_label)
CLASM Loss  = NLL(softmax(MIL_pool(logits2)), class_label)
Text Reg    = -cos(text_features[normal], text_features[others])  (拉远 normal 与异常类)
    │
    ▼ total_loss = CLAS2 + α·CLASM + txtreg_weight·TextReg
    │
    ▼ backward → clip_grad → optimizer.step (MultiStepLR, milestones=[4,8], gamma=0.1)
    │
    ▼ 每 epoch 评估 test_loader + gt → AUC/AP
    │
    ▼ 保存 best_model.pth 到 checkpoints/
```

---

## 5. 关键约束与扩展点

### 5.1 不可变约束

| 约束 | 原因 |
|------|------|
| `cuda.amp` 必须为 `false` | SVLA 与 `F.binary_cross_entropy` 在 AMP 下冲突 |
| `num_workers ≤ 2` + `persistent_workers=True` | Windows 共享内存限制（error 1455） |
| `models/svla.py` 禁止 in-place 操作 | `_mask_row_normalize` 等 in-place 会破坏 autograd |
| `text_list` 必须来自 `get_prompt_text(label_map)` | 推理与训练的类别索引必须对齐，错位会导致全错 |
| `process_feat(features, visual_length)` | 模型期望固定长度输入，必须 pad/uniform_extract 到 `visual_length` |
| CLIP 参数冻结 | 训练时仅微调 SVLA 适配层，冻结 CLIP 降低显存 |

### 5.2 扩展点

| 想做什么 | 改哪里 |
|----------|--------|
| 新增有害类别 | `configs/default.yaml` 的 `label_map` + `labels` + `text_prompts` + `models/classifier.py` 的 `LABEL_MAP` + `pipeline/alert.py` 的 `CATEGORY_DESCRIPTIONS`/`CATEGORY_ZH` |
| 替换为自有 HTTP 后端 | 在 `main.py` 新增子命令包装 `NSFWDetector.detect()`，参考 [API.md §4](API.md#4-python-api) |
| 自定义预警规则 | 重写 `AlertGenerator._determine_level()` 或 `_generate_summary()`，输入输出契约不变 |
| 替换可视化前端 | `demo/visualize.py` 是纯 Plotly，可直接重写 `create_anomaly_curve` 等 |
| 支持新视频格式 | `pipeline/inference.py` 的 `supported_formats` + `pipeline/preprocess.py` 解码逻辑 |
| ONNX 部署 | `python main.py export --format onnx`，注意当前 dummy_input 未传 padding_mask/text，部署时需自行补齐 |

---

## 6. 测试

[tests/](../tests/) 下按模块拆分：

| 文件 | 覆盖 |
|------|------|
| `test_preprocess.py` | 视频预处理 |
| `test_feature_extractor.py` | CLIP 特征提取 |
| `test_inference.py` | NSFWDetector 端到端 |
| `test_alert.py` | AlertGenerator |
| `test_metrics.py` | AUC/AP |

运行：

```bash
pytest tests/ -v -m "not slow and not gpu"
```
