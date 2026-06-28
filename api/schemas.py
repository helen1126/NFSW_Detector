"""API 请求/响应数据模型（Pydantic）。

与 pipeline.alert.AlertReport / pipeline.inference.DetectionResult 字段对齐，
供 FastAPI 自动生成 OpenAPI 文档与校验。
"""
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class HarmfulContentDetailSchema(BaseModel):
    """单类有害内容详情"""
    category_en: str = Field(..., description="类别英文名", example="Smoke")
    category_zh: str = Field(..., description="类别中文名", example="吸烟")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度 [0,1]", example=0.71)
    time_segments: str = Field(..., description="命中时间段字符串，多段以 '; ' 分隔", example="0.0-2.5s; 8.1-10.3s")
    keyframe_url: Optional[str] = Field(None, description="关键帧访问 URL（相对路径）", example="/api/v1/keyframes/keyframe_test_0.0_2.5.jpg?report_id=xxx")
    description: str = Field(..., description="中文描述", example="检测到疑似吸烟行为，可能违反平台内容规范")


class HarmfulSegmentSchema(BaseModel):
    """有害时间段"""
    start_time: float = Field(..., description="起始时间（秒）", example=0.0)
    end_time: float = Field(..., description="结束时间（秒）", example=2.5)
    score: float = Field(..., ge=0.0, le=1.0, description="段峰值异常分数", example=0.91)
    category: str = Field(..., description="类别中文名", example="吸烟")
    category_en: str = Field(..., description="类别英文名", example="Smoke")


class DetectionResultSchema(BaseModel):
    """检测结果原始字段"""
    video_id: str = Field(..., description="视频标识", example="test")
    duration: float = Field(..., description="视频时长（秒）", example=15.42)
    is_harmful: bool = Field(..., description="是否判定为有害", example=True)
    anomaly_score: float = Field(..., ge=0.0, le=1.0, description="视频级异常分数", example=0.8723)
    predicted_categories: List[str] = Field(..., description="命中阈值的类别英文名（降序）", example=["Smoke", "Blood"])
    category_scores: Dict[str, float] = Field(..., description="各类条件置信度", example={"Smoke": 0.71, "Blood": 0.12})
    harmful_segments: List[HarmfulSegmentSchema] = Field(..., description="有害时间段列表")
    keyframe_urls: List[str] = Field(..., description="关键帧访问 URL 列表", example=[])
    # 推理增强字段（可选，向后兼容）
    calibrated_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="校准后的异常分数（真实概率）", example=0.82)
    ood_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="OOD 分数，越高越像分布外内容", example=0.15)
    is_ood: Optional[bool] = Field(False, description="是否判为分布外内容", example=False)
    extra_category_info: Optional[Dict[str, Dict]] = Field(None, description="零样本扩展类别的元信息（zh 名等）", example={})


class AlertReportSchema(BaseModel):
    """预警报告（前端展示主结构）"""
    report_id: str = Field(..., description="报告唯一 ID（UUID4）", example="9f1c2a3b-4d5e-6789-abcd-ef0123456789")
    video_id: str = Field(..., description="视频标识", example="test")
    video_path: str = Field(..., description="视频文件名", example="test.mp4")
    scan_time: str = Field(..., description="扫描时间", example="2026-06-25 21:10:00")
    alert_level: str = Field(..., description="预警等级 SAFE|LOW|MEDIUM|HIGH", example="HIGH")
    anomaly_score: float = Field(..., ge=0.0, le=1.0, description="异常分数", example=0.8723)
    harmful_contents: List[HarmfulContentDetailSchema] = Field(..., description="有害内容详情列表")
    summary: str = Field(..., description="中文摘要", example="检测到高风险内容，异常评分 0.87，涉及类别：吸烟")
    action_suggestion: str = Field(..., description="处置建议", example="建议立即下架并转人工审核，等待进一步处理")
    processing_time: float = Field(..., ge=0.0, description="报告生成耗时（秒）", example=0.003)


class DetectResponseSchema(BaseModel):
    """检测接口响应（含原始检测结果与预警报告）"""
    detection: DetectionResultSchema
    report: AlertReportSchema


class CategorySchema(BaseModel):
    """类别信息"""
    id: int = Field(..., description="类别 ID", example=0)
    category_en: str = Field(..., description="英文名", example="Smoke")
    category_zh: str = Field(..., description="中文名", example="吸烟")
    description: str = Field(..., description="描述", example="检测到疑似吸烟行为，可能违反平台内容规范")


class CategoriesResponseSchema(BaseModel):
    """类别列表响应"""
    categories: List[CategorySchema]
    alert_levels: Dict[str, List[float]] = Field(..., description="预警等级阈值 [low, high)", example={"SAFE": [0, 0.3], "LOW": [0.3, 0.5]})


class HealthResponseSchema(BaseModel):
    """健康检查响应"""
    status: str = Field(..., example="ok")
    model_loaded: bool = Field(..., example=True)
    device: str = Field(..., example="cuda")
    version: str = Field(..., example="1.0.0")


class ErrorResponseSchema(BaseModel):
    """错误响应"""
    error: str = Field(..., description="错误类型", example="ValueError")
    detail: str = Field(..., description="错误详情", example="Unsupported video format: .txt")
