# NFSW Detector 前端对接 API 文档

面向前端/客户端开发者的接口规范。本文档定义视频检测的输入输出结构、字段语义、JSON 报告格式与对接方式，与代码中 `pipeline/inference.py`、`pipeline/alert.py` 保持一致。

- 适用版本：当前主干
- 维护者：NFSW Detector 项目组
- 关联文档：[README.md](../README.md) · [INSTALL.md](../INSTALL.md) · [ARCHITECTURE.md](ARCHITECTURE.md)

---

## 1. 调用流程概览

```
[视频文件]
    │  POST /detect (Gradio API)  或  NSFWDetector.detect(path)
    ▼
[VideoPreprocessor]   解码、采样、抽帧
    │
[CLIPFeatureExtractor]  视觉特征提取
    │
[SVLA 模型]  logits1=异常分数, logits2=多类别分布
    │
[DetectionResult]  段级分数 + 类别分数 + 有害时间段 + 关键帧
    │
[AlertGenerator]  规则化整理、生成摘要与处置建议
    │
[AlertReport]  最终面向用户展示的预警报告
    │
[export_json / export_html]  持久化为 JSON/HTML
```

前端两种典型对接方式：

| 方式 | 适用场景 | 入口 |
|------|----------|------|
| **Gradio Web UI** | 演示/内部审核后台 | `python main.py demo --checkpoint ...` 后访问 `http://localhost:7860` |
| **Python API + 自建后端** | 接入自有前端/服务 | `from pipeline.inference import NSFWDetector` |

> 当前未提供独立 HTTP REST 后端，前端默认通过 Gradio 提供的 API 端点调用。

---

## 2. CLI 命令

入口：`main.py`

| 命令 | 说明 | 关键参数 |
|------|------|----------|
| `train` | 训练模型 | `--config`, `--resume`, `--device` |
| `evaluate` | 评估模型，输出 AUC/AP | `--config`, `--checkpoint`, `--output` |
| `detect` | 单视频检测并生成报告 | `--config`, `--checkpoint`, `--video`, `--threshold`, `--output`, `--save-frames` |
| `demo` | 启动 Gradio Demo 服务 | `--config`, `--checkpoint`, `--port`, `--share` |
| `export` | 导出 ONNX/TorchScript | `--checkpoint`, `--format onnx\|torchscript`, `--output` |

示例：

```bash
# 单视频检测（生成 report.json）
python main.py detect --config configs/default.yaml \
  --checkpoint checkpoints/best_model.pth \
  --video test.mp4 \
  --output results/detection

# 启动 Demo（默认端口 7860）
python main.py demo --checkpoint checkpoints/best_model.pth --port 7860
```

---

## 3. Gradio API 调用

启动 `python main.py demo` 后，Gradio 自动暴露 API 端点。前端可通过 HTTP 调用：

### 3.1 端点信息

- **Base URL**: `http://localhost:7860`
- **API 文档**: `http://localhost:7860/?view=api`（Gradio 自带）
- **主要函数**: `detect_video`

### 3.2 请求示例（上传视频并触发检测）

```bash
# 1. 上传视频文件
curl -X POST http://localhost:7860/upload \
  -F "files=@test.mp4"

# 2. 调用 detect_video
curl -X POST http://localhost:7860/run/detect_video \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"path": "/tmp/gradio/xxxx/test.mp4", "url": "...", "orig_name": "test.mp4", "size": 123456, "mime_type": "video/mp4"},
      "configs/default.yaml",
      "checkpoints/best_model.pth"
    ]
  }'
```

### 3.3 响应结构

`detect_video` 返回 6 个输出，顺序与 Gradio UI 组件绑定：

| 索引 | 字段名 | 类型 | 说明 |
|------|--------|------|------|
| 0 | `overview` | string (Markdown) | 检测概览：预警等级、异常分数、摘要、处置建议、类别分数条形图 |
| 1 | `anomaly_plot` | Plotly Figure (JSON) | 异常分数时间曲线 |
| 2 | `timeline_info` | string (Markdown) | 时长与有害段数信息 |
| 3 | `keyframes` | array<`{image: filepath, caption: string}`> | 关键帧画廊 |
| 4 | `report_html` | string (HTML) | 嵌入式 HTML 预警报告 |
| 5 | `status` | string (Markdown) | 处理耗时 |

> Plotly Figure 为标准 Plotly JSON，前端可用 `plotly.js` 直接渲染。

### 3.4 Python 客户端示例

```python
from gradio_client import Client, handle_file

client = Client("http://localhost:7860")
result = client.predict(
    video_path=handle_file("test.mp4"),
    config_path="configs/default.yaml",
    checkpoint_path="checkpoints/best_model.pth",
    api_name="/detect_video"
)
overview_md, anomaly_fig, timeline_md, keyframes, report_html, status = result
```

---

## 4. Python API

### 4.1 NSFWDetector

```python
from pipeline.inference import NSFWDetector

detector = NSFWDetector(config, checkpoint_path="checkpoints/best_model.pth")
result = detector.detect("test.mp4")           # -> DetectionResult
results = detector.detect_batch(["a.mp4", ...]) # -> List[DetectionResult]
```

| 方法 | 入参 | 返回 | 说明 |
|------|------|------|------|
| `__init__(config, checkpoint_path=None)` | `config: dict`, `checkpoint_path: str\|None` | `NSFWDetector` | 不传 `checkpoint_path` 可用于纯特征提取；传则加载模型并设为 eval |
| `detect(video_path)` | `video_path: str` | `DetectionResult` | 端到端检测单视频 |
| `detect_batch(video_paths)` | `video_paths: List[str]` | `List[DetectionResult]` | 批量检测，单条失败会被跳过并告警 |

**异常**：
- `FileNotFoundError`: 视频文件不存在
- `ValueError`: 视频格式不在支持列表 `{mp4, avi, mov, mkv, wmv, flv, webm}` 内
- `RuntimeError`: GPU OOM（自动回退到半 batch 重试，仍失败则抛出）

### 4.2 AlertGenerator

```python
from pipeline.alert import AlertGenerator

alert_gen = AlertGenerator(config)
report = alert_gen.generate(result)             # -> AlertReport
reports = alert_gen.generate_batch([result, ...]) # -> List[AlertReport]
alert_gen.export_json(report, "report.json")
alert_gen.export_html(report, "report.html")
alert_gen.export_csv(reports, "reports.csv")
```

| 方法 | 说明 |
|------|------|
| `generate(detection_result)` | 接收 `DetectionResult` 或 `dict`，生成 `AlertReport` |
| `generate_batch(detection_results)` | 批量生成 |
| `export_json(report, path)` | 导出 JSON（`utf-8`，缩进 2） |
| `export_html(report, path)` | 导出独立 HTML 报告（含 base64 内联关键帧） |
| `export_csv(reports, path)` | 批量导出 CSV（`utf-8-sig`） |

---

## 5. 数据结构：DetectionResult

定义位置：`pipeline/inference.py`。前端展示检测原始数据时使用。

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `video_path` | string | 视频文件绝对路径 | `"F:/data/test.mp4"` |
| `video_id` | string | 视频标识（去扩展名的文件名） | `"test"` |
| `duration` | float | 视频时长（秒） | `15.42` |
| `is_harmful` | boolean | 是否判定为有害（`anomaly_score >= threshold`） | `true` |
| `anomaly_score` | float | 视频级异常分数 `[0,1]`，取所有段最大值 | `0.8723` |
| `predicted_categories` | string[] | 命中阈值的类别英文名（首字母大写），按分数降序 | `["Smoke", "Blood"]` |
| `category_scores` | object<string,float> | 7 类的条件置信度 `anomaly_score × P(cat\|anomaly)` | `{"Smoke": 0.71, "Blood": 0.12, ...}` |
| `segment_scores` | number[] / number[][] | 段级异常分数（sigmoid 后） | `[0.12, 0.85, 0.91, ...]` |
| `harmful_segments` | `HarmfulSegment[]` | 超阈值的有害时间段列表 | 见下表 |
| `keyframe_paths` | string[] | 关键帧图片绝对路径列表 | `["F:/data/.nsfw_keyframes/kf_..._0.0_2.5.jpg"]` |
| `attention_weights` | number[] | 注意力权重（当前实现为占位 0 数组，长度=段数） | `[0, 0, ...]` |
| `processing_time` | float | 端到端处理耗时（秒） | `3.21` |
| `detection_time` | string | 检测时间戳 `%Y-%m-%d %H:%M:%S` | `"2026-06-25 21:10:00"` |

### 5.1 HarmfulSegment

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `start_time` | float | 段起始时间（秒） | `0.0` |
| `end_time` | float | 段结束时间（秒） | `2.5` |
| `score` | float | 段峰值异常分数 | `0.91` |
| `category` | string | 类别中文名 | `"吸烟"` |
| `category_en` | string | 类别英文名 | `"Smoke"` |

### 5.2 类别分数计算说明（前端展示提示）

`category_scores` 不是简单的 softmax 概率，而是：

```
category_scores[cat_i] = anomaly_score × P(cat_i | anomaly)
P(cat_i | anomaly) = softmax(logits2)[i] / (1 - softmax(logits2)[normal])
```

- 取异常分数最高那一帧的类别分布（而非所有帧的最大值）
- 类别分数**总和不超过 anomaly_score**
- 正常视频（anomaly_prob 极小）所有类别分数会归零，避免 SAFE 视频显示误导性的高分
- `predicted_categories` 仅保留 `score >= threshold` 的类别

---

## 6. 数据结构：AlertReport

定义位置：`pipeline/alert.py`。这是前端展示预警报告的标准结构，也是 `export_json` 的输出格式。

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `report_id` | string | 报告唯一 ID（UUID4） | `"a1b2c3d4-..."` |
| `video_id` | string | 视频标识 | `"test"` |
| `video_path` | string | 视频文件路径 | `"F:/data/test.mp4"` |
| `scan_time` | string | 扫描时间 `%Y-%m-%d %H:%M:%S` | `"2026-06-25 21:10:00"` |
| `alert_level` | string | 预警等级枚举 `SAFE\|LOW\|MEDIUM\|HIGH` | `"HIGH"` |
| `anomaly_score` | float | 异常分数 `[0,1]` | `0.8723` |
| `harmful_contents` | `HarmfulContentDetail[]` | 有害内容详情列表 | 见下表 |
| `summary` | string | 中文摘要文本 | `"检测到高风险内容，异常评分 0.87，涉及类别：吸烟"` |
| `action_suggestion` | string | 处置建议文本 | `"建议立即下架并转人工审核，等待进一步处理"` |
| `processing_time` | float | 报告生成耗时（秒，不含检测） | `0.003` |

### 6.1 HarmfulContentDetail

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `category_en` | string | 类别英文名（首字母大写） | `"Smoke"` |
| `category_zh` | string | 类别中文名 | `"吸烟"` |
| `confidence` | float | 该类别置信度 `[0,1]` | `0.71` |
| `time_segments` | string | 命中时间段字符串，多段以 `"; "` 分隔 | `"0.0-2.5s; 8.1-10.3s"` |
| `keyframe_path` | string | 关键帧图片绝对路径（取第一张） | `"F:/data/.nsfw_keyframes/kf_...jpg"` |
| `description` | string | 中文描述 | `"检测到疑似吸烟行为，可能违反平台内容规范"` |

---

## 7. 枚举常量

### 7.1 有害内容类别

来源：`models/classifier.py` 的 `LABEL_MAP` / `pipeline/alert.py` 的 `CATEGORY_DESCRIPTIONS`。

| ID | 英文名 (category_en) | 中文名 (category_zh) | 描述 (description) |
|----|----------------------|----------------------|---------------------|
| 0 | Smoke | 吸烟 | 检测到疑似吸烟行为，可能违反平台内容规范 |
| 1 | Blood | 血腥 | 检测到血腥画面，可能引起观众不适 |
| 2 | Violent | 暴力 | 检测到暴力行为，违反平台社区准则 |
| 3 | Abusive | 辱骂 | 检测到辱骂行为，可能构成言语骚扰 |
| 4 | Sexy | 色情 | 检测到色情内容，违反平台内容政策 |
| 5 | Money | 金钱诈骗 | 检测到疑似金钱诈骗内容，存在欺诈风险 |
| 6 | Policy | 政治敏感 | 检测到政治敏感内容，可能违反相关规定 |

> 推理时的 `text_list` 顺序为 `[normal, smoke, blood, violent, abusive, sexy, money, policy]`（来自 `configs/default.yaml` 的 `label_map.values()`），索引 0 是 `normal`，1-7 对应上表 7 类。前端展示应使用 `category_en`/`category_zh` 字段，不要依赖索引。

### 7.2 预警等级

来源：`pipeline/alert.py` 的 `ALERT_LEVELS` / `ACTION_TEMPLATES`。

| 等级 | 分数范围 | 颜色建议 | 处置建议 (action_suggestion) |
|------|----------|----------|------------------------------|
| `SAFE` | `[0, 0.3)` | 绿色 `#4CAF50` | 内容正常，建议常规监控 |
| `LOW` | `[0.3, 0.5)` | 黄色 `#FFC107` | 建议标记待审，纳入审核队列等待处理 |
| `MEDIUM` | `[0.5, 0.8)` | 橙色 `#FF9800` | 建议限制推荐并人工复审，确认内容性质 |
| `HIGH` | `[0.8, 1.0]` | 红色 `#F44336` | 建议立即下架并转人工审核，等待进一步处理 |

判定逻辑：左闭右开区间，`HIGH` 包含 1.0。阈值可在 `configs/default.yaml` 的 `inference.alert_levels` 配置。

---

## 8. JSON 报告完整示例

`alert_gen.export_json(report, "report.json")` 输出结构（已格式化示例）：

```json
{
  "report_id": "9f1c2a3b-4d5e-6789-abcd-ef0123456789",
  "video_id": "test_smoke_001",
  "video_path": "F:/data/test_smoke_001.mp4",
  "scan_time": "2026-06-25 21:10:00",
  "alert_level": "HIGH",
  "anomaly_score": 0.8723,
  "harmful_contents": [
    {
      "category_en": "Smoke",
      "category_zh": "吸烟",
      "confidence": 0.71,
      "time_segments": "0.0-2.5s; 8.1-10.3s",
      "keyframe_path": "F:/data/.nsfw_keyframes/keyframe_test_smoke_001_0.0_2.5.jpg",
      "description": "检测到疑似吸烟行为，可能违反平台内容规范"
    },
    {
      "category_en": "Blood",
      "category_zh": "血腥",
      "confidence": 0.12,
      "time_segments": "0.0-2.5s; 8.1-10.3s",
      "keyframe_path": "F:/data/.nsfw_keyframes/keyframe_test_smoke_001_0.0_2.5.jpg",
      "description": "检测到血腥画面，可能引起观众不适"
    }
  ],
  "summary": "检测到高风险内容，异常评分 0.87，涉及类别：吸烟、血腥",
  "action_suggestion": "建议立即下架并转人工审核，等待进一步处理",
  "processing_time": 0.0032
}
```

### 8.1 安全视频示例（无有害内容）

```json
{
  "report_id": "...",
  "video_id": "test_normal_001",
  "video_path": "F:/data/test_normal_001.mp4",
  "scan_time": "2026-06-25 21:12:00",
  "alert_level": "SAFE",
  "anomaly_score": 0.21,
  "harmful_contents": [],
  "summary": "视频内容安全，未检测到违规内容",
  "action_suggestion": "内容正常，建议常规监控",
  "processing_time": 0.0028
}
```

---

## 9. 前端对接建议

### 9.1 推荐组件结构

```
检测结果页
├── 顶部状态条
│   ├── 等级徽章（color 由 alert_level 映射，见 §7.2）
│   ├── 异常分数（anomaly_score，保留 4 位小数）
│   ├── 处置建议（action_suggestion）
│   └── 处理耗时（processing_time）
├── 摘要卡片（summary）
├── 类别分数区（harmful_contents 或 DetectionResult.category_scores）
│   └── 进度条（confidence × 100%，超阈值红色）
├── 时间轴分析
│   ├── Plotly 异常曲线（anomaly_plot，直接渲染 Plotly JSON）
│   ├── 视频播放器
│   └── 有害时间段列表（time_segments 分号分隔，需解析为多段）
├── 关键帧画廊（keyframes，img src 用 file:// 或经后端代理）
└── 报告下载（report_html / JSON 导出）
```

### 9.2 字段渲染约定

| 字段 | 渲染建议 |
|------|----------|
| `anomaly_score` | 保留 4 位小数；可用仪表盘组件，刻度按 §7.2 着色 |
| `alert_level` | 徽章组件，颜色见 §7.2 |
| `category_scores` / `harmful_contents[].confidence` | 横向进度条，`>= 0.5` 标红 |
| `time_segments` | 字符串 `"0.0-2.5s; 8.1-10.3s"`，按 `; ` 分割后逐段渲染 |
| `keyframe_path` / `keyframe_paths` | 本地路径，前端需通过后端文件接口转换成 URL；`export_html` 已 base64 内联 |
| `predicted_categories` | Tag/Chip 组件，按 `category_scores` 降序展示 |
| `summary` / `action_suggestion` | 直接文本展示，`action_suggestion` 在 SAFE 等级下可能为空字符串 |

### 9.3 Plotly 图表对接

`anomaly_plot` 是 Plotly Figure 对象的 JSON 序列化形式，前端：

```javascript
// 假设从 Gradio API 拿到 anomaly_plot (对象)
Plotly.newPlot('anomaly-plot', anomaly_plot.data, anomaly_plot.layout);
```

横轴为段索引（非真实时间），纵轴为 `[0,1]` 异常分数，红色虚线为 `threshold`。

### 9.4 关键帧展示

`DetectionResult.keyframe_paths` 为本地绝对路径。前端展示有两种方案：

1. **同源后端代理**（推荐）：自建 FastAPI/Flask 暴露 `GET /keyframes/{filename}` 接口读取文件返回 `image/jpeg`
2. **base64 内联**：使用 `alert_gen.export_html(report)` 生成的报告已内联 base64，可直接 iframe 嵌入

> 直接将本地路径塞进 `<img src="...">` 在浏览器中不可用，受 `file://` 协议限制。

---

## 10. 错误处理与边界情况

| 场景 | 行为 | 前端处理建议 |
|------|------|--------------|
| 视频文件不存在 | `detect()` 抛 `FileNotFoundError` | 显示 "文件不存在" 提示，让用户重新上传 |
| 视频格式不支持 | `detect()` 抛 `ValueError` | 显示 "支持格式：mp4/avi/mov/mkv/wmv/flv/webm" |
| GPU OOM | 自动半 batch 重试，仍失败抛 `RuntimeError` | 提示 "显存不足，请缩短视频或使用 CPU" |
| 长视频 | 自动降低采样率并告警 | 检测仍会完成，展示时提示 "已降采样" |
| 无 logits2（仅二分类） | `category_scores` 全部为 0，`predicted_categories` 为空 | 隐藏类别分数区域，仅展示异常分数 |
| 安全视频 | `harmful_contents` 为空数组，`summary` 显示 "未检测到违规内容" | 关键帧区域显示 "无关键帧" |
| `anomaly_score` 在阈值附近 | `is_harmful` 与人工判断可能不符 | UI 标注 " borderline" 状态，提示人工复审 |

---

## 11. 配置参考

`configs/default.yaml` 中与前端/对接相关的配置段：

```yaml
inference:
  anomaly_threshold: 0.5       # 二分类判定阈值，影响 is_harmful 与 predicted_categories
  alert_levels:                # 预警等级阈值（左闭右开，HIGH 包含 1.0）
    high: 0.8
    medium: 0.5
    low: 0.3
  max_duration: 300            # 视频最大时长（秒），超过会降采样

demo:
  port: 7860                   # Gradio 服务端口
  share: false                 # 是否生成公网链接
  max_file_size: 500           # 上传文件大小上限（MB）
```

---

## 12. 变更日志

| 日期 | 变更 |
|------|------|
| 2026-06-25 | 初版：定义 DetectionResult/AlertReport 字段、Gradio API、JSON 报告示例、前端对接建议 |
