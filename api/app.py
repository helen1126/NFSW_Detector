"""NFSW Detector FastAPI 应用。

提供 RESTful API 用于视频有害内容检测与预警报告查询。
复用 pipeline.inference.NSFWDetector 与 pipeline.alert.AlertGenerator。

启动：
    python main.py serve --checkpoint checkpoints/best_model.pth
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

文档：
    Swagger UI:  /docs
    ReDoc:       /redoc
    OpenAPI JSON: /openapi.json
"""
import os
import json
import shutil
import tempfile
from typing import Dict, Optional
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from pipeline.inference import NSFWDetector
from pipeline.alert import AlertGenerator, AlertReport, CATEGORY_DESCRIPTIONS
from models.classifier import LABEL_MAP
from api.schemas import (
    DetectResponseSchema,
    AlertReportSchema,
    CategoriesResponseSchema,
    HealthResponseSchema,
    ErrorResponseSchema,
)

API_VERSION = "1.0.0"
SUPPORTED_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


class AppState:
    """应用全局状态：模型与报告存储"""

    def __init__(self):
        self.detector: Optional[NSFWDetector] = None
        self.alert_gen: Optional[AlertGenerator] = None
        self.config: dict = {}
        self.config_path: str = ""
        self.checkpoint_path: str = ""
        # report_id -> {"report": dict, "keyframes": {filename: abspath}}
        self.reports: Dict[str, dict] = {}
        self.reports_dir: str = "reports/api"


state = AppState()


def load_app_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_state(config_path: str, checkpoint_path: Optional[str]):
    """初始化全局状态（加载模型）。在 lifespan 或 main.py serve 中调用。"""
    state.config_path = config_path
    state.config = load_app_config(config_path)
    state.checkpoint_path = checkpoint_path or ""
    os.makedirs(state.reports_dir, exist_ok=True)
    if checkpoint_path:
        state.detector = NSFWDetector(state.config, checkpoint_path=checkpoint_path)
        state.detector.model.eval()
    else:
        state.detector = None
    state.alert_gen = AlertGenerator(state.config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时按环境变量加载模型（默认不加载，由 main.py serve 显式初始化）。"""
    if state.detector is None:
        cfg_path = os.environ.get("NSFW_CONFIG", "configs/default.yaml")
        ckpt_path = os.environ.get("NSFW_CHECKPOINT") or None
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                init_state(cfg_path, ckpt_path)
            except Exception as e:
                print(f"[WARN] 模型加载失败，API 将以无模型模式启动: {e}")
    yield


app = FastAPI(
    title="NFSW Detector API",
    description=(
        "多模态视频有害内容检测与预警 RESTful API。\n\n"
        "基于 SVLA（Shot-Conditioned Vision-Language Adaptation）模型，"
        "支持上传视频进行端到端检测，返回异常分数、类别置信度、有害时间段、"
        "关键帧与结构化预警报告。\n\n"
        "**主要能力：**\n"
        "* 视频有害内容检测（7 类异常 + 二分类异常分数）\n"
        "* 预警等级判定（SAFE/LOW/MEDIUM/HIGH）\n"
        "* 关键帧抽取与图片服务\n"
        "* 结构化报告查询与 JSON 持久化\n\n"
        "关联文档：[docs/API.md](../docs/API.md)"
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_keyframe_url(report_id: str, abspath: str) -> str:
    """构建关键帧访问 URL。"""
    filename = os.path.basename(abspath)
    return f"/api/v1/keyframes/{filename}?report_id={report_id}"


def _serialize_detection(result, report_id: str) -> dict:
    """将 DetectionResult 转换为响应字典，关键帧路径转为 URL。"""
    keyframe_urls = [_build_keyframe_url(report_id, p) for p in result.keyframe_paths]
    return {
        "video_id": result.video_id,
        "duration": result.duration,
        "is_harmful": result.is_harmful,
        "anomaly_score": result.anomaly_score,
        "predicted_categories": result.predicted_categories,
        "category_scores": {k: float(v) for k, v in result.category_scores.items()},
        "harmful_segments": [
            {
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "score": seg.score,
                "category": seg.category,
                "category_en": seg.category_en,
            }
            for seg in result.harmful_segments
        ],
        "keyframe_urls": keyframe_urls,
    }


def _serialize_report(report: AlertReport, report_id: str) -> dict:
    """将 AlertReport 转换为响应字典，注入 keyframe_url。"""
    from dataclasses import asdict
    data = asdict(report)
    # 为每个 harmful_content 注入 keyframe_url
    keyframe_map = state.reports.get(report_id, {}).get("keyframes", {})
    for hc in data["harmful_contents"]:
        kp = hc.get("keyframe_path", "")
        fname = os.path.basename(kp) if kp else ""
        if fname and fname in keyframe_map:
            hc["keyframe_url"] = _build_keyframe_url(report_id, keyframe_map[fname])
        else:
            hc["keyframe_url"] = None
    return data


def _persist_report(report: AlertReport):
    """持久化报告到 reports/api/{report_id}.json。"""
    from dataclasses import asdict
    path = os.path.join(state.reports_dir, f"{report.report_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, ensure_ascii=False, indent=2, default=str)


# ==================== 路由 ====================

@app.get("/", tags=["元信息"], summary="API 根信息")
async def root():
    """返回 API 基本信息。"""
    return {
        "name": "NFSW Detector API",
        "version": API_VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json",
        "endpoints": [
            "GET /api/v1/health",
            "GET /api/v1/categories",
            "POST /api/v1/detect",
            "GET /api/v1/reports/{report_id}",
            "GET /api/v1/keyframes/{filename}",
        ],
    }


@app.get(
    "/api/v1/health",
    response_model=HealthResponseSchema,
    tags=["系统"],
    summary="健康检查",
    description="探测服务与模型加载状态，前端可用于连通性检测。",
)
async def health():
    device = str(state.detector.device) if state.detector else "cpu"
    return HealthResponseSchema(
        status="ok",
        model_loaded=state.detector is not None and state.detector.model is not None,
        device=device,
        version=API_VERSION,
    )


@app.get(
    "/api/v1/categories",
    response_model=CategoriesResponseSchema,
    tags=["元数据"],
    summary="获取类别与预警等级",
    description="返回 7 类有害内容定义、中文描述及预警等级阈值，供前端渲染类别选择器与等级配色。",
)
async def get_categories():
    from pipeline.alert import ALERT_LEVELS
    categories = []
    for cid, info in LABEL_MAP.items():
        cat_en = info["en"]
        categories.append({
            "id": cid,
            "category_en": cat_en,
            "category_zh": info["zh"],
            "description": CATEGORY_DESCRIPTIONS.get(cat_en.lower(), f"检测到{info['zh']}内容"),
        })
    alert_levels = {k: [v[0], v[1]] for k, v in ALERT_LEVELS.items()}
    return CategoriesResponseSchema(categories=categories, alert_levels=alert_levels)


@app.post(
    "/api/v1/detect",
    response_model=DetectResponseSchema,
    responses={
        400: {"model": ErrorResponseSchema, "description": "视频格式不支持或文件无效"},
        500: {"model": ErrorResponseSchema, "description": "检测过程内部错误"},
        503: {"model": ErrorResponseSchema, "description": "模型未加载"},
    },
    tags=["检测"],
    summary="上传视频并检测",
    description=(
        "上传视频文件执行端到端有害内容检测。返回检测结果（含段级分数、类别置信度、"
        "有害时间段、关键帧 URL）与结构化预警报告。\n\n"
        "**支持的格式：** mp4, avi, mov, mkv, wmv, flv, webm\n\n"
        "**阈值覆盖：** 可选 `threshold` 参数临时覆盖 `inference.anomaly_threshold`。"
    ),
)
async def detect_video(
    file: UploadFile = File(..., description="视频文件"),
    threshold: float = Query(None, ge=0.0, le=1.0, description="临时异常阈值，覆盖配置默认值"),
):
    if state.detector is None:
        raise HTTPException(status_code=503, detail="模型未加载，请通过 --checkpoint 启动或设置 NSFW_CHECKPOINT 环境变量")

    filename = file.filename or "upload.mp4"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail=f"不支持的视频格式: {ext}，支持: {sorted(SUPPORTED_FORMATS)}")

    # 保存上传文件到临时路径
    tmp_dir = tempfile.mkdtemp(prefix="nsfw_upload_")
    tmp_path = os.path.join(tmp_dir, filename)
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 临时覆盖阈值
        original_threshold = None
        if threshold is not None:
            original_threshold = state.config.get("inference", {}).get("anomaly_threshold")
            state.config.setdefault("inference", {})["anomaly_threshold"] = threshold
            state.detector.threshold = threshold

        try:
            result = state.detector.detect(tmp_path)
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=f"推理失败: {e}")
        finally:
            # 恢复阈值
            if original_threshold is not None:
                state.config["inference"]["anomaly_threshold"] = original_threshold
                state.detector.threshold = original_threshold

        # 生成报告
        report = state.alert_gen.generate(result)

        # 存储关键帧映射
        keyframe_map = {os.path.basename(p): p for p in result.keyframe_paths}
        state.reports[report.report_id] = {
            "report": report,
            "keyframes": keyframe_map,
        }
        _persist_report(report)

        detection_data = _serialize_detection(result, report.report_id)
        report_data = _serialize_report(report, report.report_id)
        return DetectResponseSchema(detection=detection_data, report=report_data)
    finally:
        # 清理上传的临时视频文件（关键帧已由 detector 保存到独立目录，不受影响）
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get(
    "/api/v1/reports/{report_id}",
    response_model=AlertReportSchema,
    responses={
        404: {"model": ErrorResponseSchema, "description": "报告不存在"},
    },
    tags=["报告"],
    summary="获取预警报告",
    description="按 report_id 获取已持久化的预警报告。报告在 POST /detect 时生成并保存到 reports/api/ 目录。",
)
async def get_report(report_id: str):
    # 优先从内存取
    entry = state.reports.get(report_id)
    if entry is not None:
        return _serialize_report(entry["report"], report_id)

    # 内存未命中则尝试从磁盘加载
    path = os.path.join(state.reports_dir, f"{report_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"报告不存在: {report_id}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 磁盘加载的报告无 keyframe_map，keyframe_url 置空
    for hc in data.get("harmful_contents", []):
        hc["keyframe_url"] = None
    return data


@app.get(
    "/api/v1/keyframes/{filename}",
    responses={
        404: {"model": ErrorResponseSchema, "description": "关键帧不存在"},
    },
    tags=["报告"],
    summary="获取关键帧图片",
    description="按文件名与 report_id 获取关键帧图片。filename 与 report_id 均来自 /detect 或 /reports 响应。",
)
async def get_keyframe(
    filename: str,
    report_id: str = Query(..., description="所属报告 ID"),
):
    entry = state.reports.get(report_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"报告不存在或已过期: {report_id}")
    abspath = entry["keyframes"].get(filename)
    if abspath is None or not os.path.exists(abspath):
        raise HTTPException(status_code=404, detail=f"关键帧不存在: {filename}")
    return FileResponse(abspath, media_type="image/jpeg")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.__class__.__name__, "detail": exc.detail},
    )
