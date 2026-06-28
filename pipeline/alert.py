import os
import json
import uuid
import base64
import time
import csv
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
from datetime import datetime

ALERT_LEVELS = {
    "SAFE": (0, 0.3),
    "LOW": (0.3, 0.5),
    "MEDIUM": (0.5, 0.8),
    "HIGH": (0.8, 1.0),
}

CATEGORY_DESCRIPTIONS = {
    "Smoke": "检测到疑似吸烟行为，可能违反平台内容规范",
    "Blood": "检测到血腥画面，可能引起观众不适",
    "Violent": "检测到暴力行为，违反平台社区准则",
    "Abusive": "检测到辱骂行为，可能构成言语骚扰",
    "Sexy": "检测到色情内容，违反平台内容政策",
    "Money": "检测到疑似金钱诈骗内容，存在欺诈风险",
    "Policy": "检测到政治敏感内容，可能违反相关规定",
}

ACTION_TEMPLATES = {
    "HIGH": "建议立即下架并转人工审核，等待进一步处理",
    "MEDIUM": "建议限制推荐并人工复审，确认内容性质",
    "LOW": "建议标记待审，纳入审核队列等待处理",
    "SAFE": "内容正常，建议常规监控",
}

CATEGORY_ZH = {
    "Smoke": "吸烟",
    "Blood": "血腥",
    "Violent": "暴力",
    "Abusive": "辱骂",
    "Sexy": "色情",
    "Money": "金钱诈骗",
    "Policy": "政治敏感",
}


@dataclass
class HarmfulContentDetail:
    category_en: str
    category_zh: str
    confidence: float
    time_segments: str
    keyframe_path: str
    description: str


@dataclass
class AlertReport:
    report_id: str
    video_id: str
    video_path: str
    scan_time: str
    alert_level: str
    anomaly_score: float
    harmful_contents: List[HarmfulContentDetail]
    summary: str
    action_suggestion: str
    processing_time: float


class AlertGenerator:
    def __init__(self, config=None):
        self.config = config or {}

    def generate(self, detection_result) -> AlertReport:
        start = time.time()
        dr = self._normalize_input(detection_result)
        anomaly_score = float(dr.get("anomaly_score", 0.0))
        alert_level = self._determine_level(anomaly_score)
        harmful_contents = self._extract_harmful_contents(dr)
        summary = self._generate_summary(dr)
        action_suggestion = ACTION_TEMPLATES.get(alert_level, "")
        processing_time = time.time() - start

        report = AlertReport(
            report_id=str(uuid.uuid4()),
            video_id=dr.get("video_id", ""),
            video_path=dr.get("video_path", ""),
            scan_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            alert_level=alert_level,
            anomaly_score=anomaly_score,
            harmful_contents=harmful_contents,
            summary=summary,
            action_suggestion=action_suggestion,
            processing_time=processing_time,
        )
        return report

    def _normalize_input(self, detection_result) -> dict:
        """将 dataclass 或 dict 输入统一转换为字典格式"""
        if isinstance(detection_result, dict):
            return detection_result
        dr = asdict(detection_result)
        # 适配字段名：DetectionResult.predicted_categories -> categories
        dr["categories"] = dr.get("predicted_categories", [])
        # 从 harmful_segments 提取 time_segments 字符串
        if dr.get("harmful_segments"):
            dr["time_segments"] = "; ".join(
                f"{seg['start_time']:.1f}-{seg['end_time']:.1f}s"
                for seg in dr["harmful_segments"]
            )
        else:
            dr["time_segments"] = ""
        # keyframe_paths (list) -> keyframe_path (string, 取第一个)
        kp_list = dr.get("keyframe_paths", [])
        dr["keyframe_path"] = kp_list[0] if kp_list else ""
        return dr

    def generate_batch(self, detection_results) -> List[AlertReport]:
        return [self.generate(result) for result in detection_results]

    def _determine_level(self, score) -> str:
        if score < 0.3:
            return "SAFE"
        elif score < 0.5:
            return "LOW"
        elif score < 0.8:
            return "MEDIUM"
        else:
            return "HIGH"

    def _generate_summary(self, detection_result) -> str:
        anomaly_score = detection_result.get("anomaly_score", 0.0)
        categories = detection_result.get("categories", [])

        if not categories:
            if anomaly_score < 0.3:
                return "视频内容安全，未检测到违规内容"
            else:
                return f"异常评分 {anomaly_score:.2f}，但未识别具体违规类别"

        category_names = []
        for cat in categories:
            cat_name = cat if isinstance(cat, str) else cat.get("category", "")
            zh = CATEGORY_ZH.get(cat_name, cat_name)
            category_names.append(zh)

        level = self._determine_level(anomaly_score)
        level_map = {"SAFE": "安全", "LOW": "低风险", "MEDIUM": "中风险", "HIGH": "高风险"}
        summary = f"检测到{level_map.get(level, '')}内容，异常评分 {anomaly_score:.2f}，涉及类别：{'、'.join(category_names)}"
        return summary

    def _extract_harmful_contents(self, detection_result) -> List[HarmfulContentDetail]:
        contents = []
        categories = detection_result.get("categories", [])

        for cat in categories:
            if isinstance(cat, str):
                cat_en = cat
                confidence = float(detection_result.get("anomaly_score", 0.0))
                time_segments = detection_result.get("time_segments", "")
                keyframe_path = detection_result.get("keyframe_path", "")
            elif isinstance(cat, dict):
                cat_en = cat.get("category", "")
                confidence = float(cat.get("confidence", 0.0))
                time_segments = cat.get("time_segments", "")
                keyframe_path = cat.get("keyframe_path", "")
            else:
                continue

            cat_zh = CATEGORY_ZH.get(cat_en, cat_en)
            description = CATEGORY_DESCRIPTIONS.get(cat_en, f"检测到{cat_zh}内容")

            contents.append(
                HarmfulContentDetail(
                    category_en=cat_en,
                    category_zh=cat_zh,
                    confidence=confidence,
                    time_segments=str(time_segments),
                    keyframe_path=str(keyframe_path),
                    description=description,
                )
            )

        return contents

    def export_json(self, report, output_path):
        def _convert(obj):
            if isinstance(obj, float):
                return float(obj)
            if hasattr(obj, "asdict"):
                return asdict(obj)
            if isinstance(obj, list):
                return [_convert(i) for i in obj]
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            return obj

        data = asdict(report)
        data = _convert(data)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def export_html(self, report, output_path):
        data = asdict(report)

        keyframe_html = ""
        for hc in data["harmful_contents"]:
            kp = hc.get("keyframe_path", "")
            if kp and os.path.isfile(kp):
                with open(kp, "rb") as img_file:
                    b64 = base64.b64encode(img_file.read()).decode("utf-8")
                keyframe_html += f'<div class="keyframe"><p>{hc["category_zh"]} ({hc["category_en"]}) - 置信度: {hc["confidence"]:.2f}</p><img src="data:image/jpeg;base64,{b64}" /></div>'

        harmful_rows = ""
        for hc in data["harmful_contents"]:
            harmful_rows += f"""<tr>
                <td>{hc['category_zh']}</td>
                <td>{hc['category_en']}</td>
                <td>{hc['confidence']:.4f}</td>
                <td>{hc['time_segments']}</td>
                <td>{hc['description']}</td>
            </tr>"""

        level_colors = {
            "SAFE": "#4CAF50",
            "LOW": "#FFC107",
            "MEDIUM": "#FF9800",
            "HIGH": "#F44336",
        }
        level_color = level_colors.get(data["alert_level"], "#999")

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>审核报告 - {data['report_id']}</title>
<style>
body {{ font-family: "Microsoft YaHei", sans-serif; margin: 20px; background: #f5f5f5; }}
.container {{ max-width: 900px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
h1 {{ color: #333; border-bottom: 2px solid {level_color}; padding-bottom: 10px; }}
.meta {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 20px 0; }}
.meta-item {{ padding: 8px; background: #f9f9f9; border-radius: 4px; }}
.meta-label {{ font-weight: bold; color: #555; }}
.level-badge {{ display: inline-block; padding: 4px 12px; border-radius: 4px; color: #fff; background: {level_color}; font-weight: bold; }}
table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
th {{ background: #f0f0f0; }}
.keyframe {{ margin: 10px 0; }}
.keyframe img {{ max-width: 400px; border-radius: 4px; }}
.summary {{ background: #fff3e0; padding: 15px; border-radius: 4px; margin: 15px 0; }}
.action {{ background: #e8f5e9; padding: 15px; border-radius: 4px; margin: 15px 0; }}
</style>
</head>
<body>
<div class="container">
<h1>内容审核报告</h1>
<div class="meta">
    <div class="meta-item"><span class="meta-label">报告ID：</span>{data['report_id']}</div>
    <div class="meta-item"><span class="meta-label">视频ID：</span>{data['video_id']}</div>
    <div class="meta-item"><span class="meta-label">视频路径：</span>{data['video_path']}</div>
    <div class="meta-item"><span class="meta-label">扫描时间：</span>{data['scan_time']}</div>
    <div class="meta-item"><span class="meta-label">告警等级：</span><span class="level-badge">{data['alert_level']}</span></div>
    <div class="meta-item"><span class="meta-label">异常评分：</span>{data['anomaly_score']:.4f}</div>
    <div class="meta-item"><span class="meta-label">处理耗时：</span>{data['processing_time']:.3f}s</div>
</div>
<div class="summary"><strong>摘要：</strong>{data['summary']}</div>
<div class="action"><strong>处理建议：</strong>{data['action_suggestion']}</div>
<h2>违规内容详情</h2>
<table>
<tr><th>类别(中)</th><th>类别(英)</th><th>置信度</th><th>时间段</th><th>描述</th></tr>
{harmful_rows}
</table>
<h2>关键帧</h2>
{keyframe_html}
</div>
</body>
</html>"""

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    def export_csv(self, reports, output_path):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        headers = [
            "report_id",
            "video_id",
            "video_path",
            "scan_time",
            "alert_level",
            "anomaly_score",
            "summary",
            "action_suggestion",
            "processing_time",
            "harmful_categories",
        ]

        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for report in reports:
                data = asdict(report)
                categories = "; ".join(
                    f"{hc['category_zh']}({hc['confidence']:.2f})"
                    for hc in data["harmful_contents"]
                )
                writer.writerow([
                    data["report_id"],
                    data["video_id"],
                    data["video_path"],
                    data["scan_time"],
                    data["alert_level"],
                    f"{data['anomaly_score']:.4f}",
                    data["summary"],
                    data["action_suggestion"],
                    f"{data['processing_time']:.3f}",
                    categories,
                ])
