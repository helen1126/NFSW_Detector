from dataclasses import dataclass, field
from typing import List, Optional, Dict
import os
import time
import json
import numpy as np
import torch

from pipeline.preprocess import VideoPreprocessor
from pipeline.feature_extractor import CLIPFeatureExtractor
from models.svla import SVLA
from models.classifier import LABEL_MAP, EN_TO_ZH
from utils.tools import process_feat, get_prompt_text


@dataclass
class HarmfulSegment:
    start_time: float
    end_time: float
    score: float
    category: str
    category_en: str


@dataclass
class DetectionResult:
    video_path: str
    video_id: str
    duration: float
    is_harmful: bool
    anomaly_score: float
    predicted_categories: List[str]
    category_scores: Dict[str, float]
    segment_scores: np.ndarray
    harmful_segments: List[HarmfulSegment]
    keyframe_paths: List[str]
    attention_weights: np.ndarray
    processing_time: float
    detection_time: str
    # 推理增强字段（默认值保证向后兼容）
    calibrated_score: float = 0.0
    ood_score: float = 0.0
    is_ood: bool = False
    extra_category_info: Dict[str, Dict] = field(default_factory=dict)


class NSFWDetector:
    def __init__(self, config, checkpoint_path=None):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.config = config
        self.preprocessor = VideoPreprocessor(config)
        # 特征提取器骨干：当训练数据是低维预提取特征（如 SVA 的 512 维 ViT-B/16），
        # 而 clip_variant 是高维骨干（如 ViT-L/14 768 维）时，必须与训练特征维度一致
        clip_variant = config.get('model', {}).get('clip_variant', 'ViT-B/16')
        extractor_variant = config.get('model', {}).get('feature_extractor_variant', clip_variant)
        self.feature_extractor = CLIPFeatureExtractor(extractor_variant, self.device)
        self.model = None
        if checkpoint_path is not None:
            self.model = self._load_model(checkpoint_path)
            self.model.eval()
        self.threshold = config['inference']['anomaly_threshold']

        # 分数校准（Isotonic Regression）
        from pipeline.calibration import ScoreCalibrator
        cal_cfg = config.get('calibration', {})
        self.calibrator = None
        if cal_cfg.get('enabled', False):
            cal_path = cal_cfg.get('path', 'checkpoints/calibrator.pkl')
            self.calibrator = ScoreCalibrator.load(cal_path)
            if not self.calibrator.fitted:
                import warnings
                warnings.warn(f"校准器未拟合或加载失败 ({cal_path})，将返回原始分数")

        # OOD 检测
        ood_cfg = config.get('ood', {})
        self.ood_enabled = ood_cfg.get('enabled', False)
        self.ood_threshold = float(ood_cfg.get('threshold', 0.5))

        # 关键帧质量加权
        fq_cfg = config.get('frame_quality', {})
        self.frame_quality_enabled = fq_cfg.get('enabled', False)

        # 零样本新类别扩展
        zs_cfg = config.get('zero_shot', {})
        self.extra_categories = {}
        if zs_cfg.get('enabled', False) and self.feature_extractor is not None:
            self._init_extra_categories(zs_cfg.get('extra_categories', {}))

    def _init_extra_categories(self, extra_categories_cfg):
        """初始化零样本扩展类别，预计算文本特征。

        Args:
            extra_categories_cfg: dict, 形如
                {"gambling": {"prompts": [...], "zh": "赌博"}, ...}
        """
        self.extra_categories = {}
        for cat_key, info in extra_categories_cfg.items():
            prompts = info.get('prompts', [])
            zh = info.get('zh', cat_key)
            if not prompts:
                continue
            text_feats = self.feature_extractor.extract_text_features(prompts)
            mean_text_feat = text_feats.mean(axis=0)
            mean_text_feat = mean_text_feat / (np.linalg.norm(mean_text_feat) + 1e-10)
            self.extra_categories[cat_key] = {
                'text_feat': mean_text_feat,
                'zh': zh,
                'prompts': prompts,
            }

    def detect(self, video_path, num_segments=None) -> DetectionResult:
        start_time = time.time()
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        ext = os.path.splitext(video_path)[1].lower()
        supported_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        if ext not in supported_formats:
            raise ValueError(f"Unsupported video format: {ext}")

        try:
            preprocess_result = self.preprocessor.preprocess(video_path, num_segments=num_segments)
        except Exception:
            preprocess_result = self.preprocessor.preprocess(
                video_path, reduced_sample_rate=True, num_segments=num_segments
            )
            import warnings
            warnings.warn(f"Long video detected, processing with reduced sample rate: {video_path}")

        frames = preprocess_result['frames']
        timestamps = preprocess_result['timestamps']
        fps = preprocess_result['fps']
        duration = preprocess_result['duration']
        video_id = os.path.splitext(os.path.basename(video_path))[0]

        batch_size = self.config.get('inference', {}).get('batch_size', 32)
        all_features = []
        try:
            for i in range(0, len(frames), batch_size):
                batch = frames[i:i + batch_size]
                features = self.feature_extractor.extract_visual_features(batch)
                all_features.append(features)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                reduced_batch = max(1, batch_size // 2)
                all_features = []
                for i in range(0, len(frames), reduced_batch):
                    batch = frames[i:i + reduced_batch]
                    features = self.feature_extractor.extract_visual_features(batch)
                    all_features.append(features)
            else:
                raise

        features = np.concatenate(all_features, axis=0)

        result = self._run_inference(
            features=features,
            video_path=video_path,
            video_id=video_id,
            duration=duration,
            fps=fps,
            timestamps=timestamps,
            raw_frames=frames,
            start_time=start_time,
        )
        return result

    def detect_image(self, image_path) -> DetectionResult:
        """图片检测：将单张图片视为 1 帧视频，复用 _run_inference。"""
        start_time = time.time()
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        ext = os.path.splitext(image_path)[1].lower()
        supported_img_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        if ext not in supported_img_formats:
            raise ValueError(f"Unsupported image format: {ext}")

        import cv2
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames = np.array([frame])  # [1, H, W, 3]

        features = self.feature_extractor.extract_visual_features(frames)  # [1, D]
        video_id = os.path.splitext(os.path.basename(image_path))[0]

        return self._run_inference(
            features=features,
            video_path=image_path,
            video_id=video_id,
            duration=0.0,
            fps=1.0,
            timestamps=[(0.0, 0.0)],
            raw_frames=frames,
            start_time=start_time,
        )

    def _run_inference(self, features, video_path, video_id,
                       duration, fps, timestamps, raw_frames=None,
                       start_time=None) -> DetectionResult:
        """从已提取的视觉特征跑到 DetectionResult。detect() 和 detect_image() 共用。

        Args:
            features: [num_frames, D] numpy 数组（process_feat 之前）
            video_path: 视频/图片路径（用于关键帧抽取）
            video_id: 标识
            duration: 时长（图片为 0.0）
            fps: 帧率（图片为 1.0）
            timestamps: [(start, end), ...]
            raw_frames: 原始帧数组（用于质量加权），可为 None
            start_time: 调用方起始时间戳（用于计算 processing_time）
        """
        if start_time is None:
            start_time = time.time()

        # 处理特征到模型期望的长度（visual_length），与训练时的 process_feat 一致
        visual_length = self.model.visual_length
        features, valid_length = process_feat(features, visual_length)

        with torch.inference_mode():
            features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
            features_tensor = features_tensor.unsqueeze(0)
            B, S, D = features_tensor.shape
            lengths = torch.tensor([valid_length], dtype=torch.long)
            padding_mask = torch.arange(S, device=self.device).unsqueeze(0) >= lengths.unsqueeze(1).to(self.device)
            text_list = get_prompt_text(self.config.get('label_map', {}), self.config.get('text_prompts', {}))
            text_features, logits1, logits2, shot_slices, attn_weights = self.model(
                features_tensor, padding_mask, text_list, lengths
            )
            segment_scores = torch.sigmoid(logits1[:, 0]).cpu().numpy()
            if logits2 is not None:
                class_probs = torch.softmax(logits2, dim=-1)  # [B, T, num_text]
                class_scores_raw = class_probs[0].cpu().numpy()  # [T, num_text]
            else:
                class_scores_raw = np.zeros((1, len(LABEL_MAP) + 1))
            attention_weights = np.zeros(S)

        # 仅在有效帧范围内计算异常分数，过滤 padding 帧干扰
        valid_scores = segment_scores.reshape(-1)[:valid_length]

        # 改动 4: 关键帧质量加权（低质量帧分数降权）
        if self.frame_quality_enabled and raw_frames is not None:
            qualities = self._compute_frame_quality(raw_frames)
            if len(qualities) >= valid_length:
                seg_qualities = qualities[:valid_length]
            else:
                seg_qualities = np.pad(qualities, (0, valid_length - len(qualities)))
            valid_scores = valid_scores * seg_qualities

        if valid_length > 0:
            anomaly_score = float(np.max(valid_scores))
        else:
            anomaly_score = 0.0

        # 改动 1: 分数校准
        calibrated_score = anomaly_score
        if self.calibrator is not None and self.calibrator.fitted:
            calibrated_score = self.calibrator.transform(anomaly_score)

        is_harmful = anomaly_score >= self.threshold

        # 改动 3: OOD 检测
        ood_score = 0.0
        is_ood = False
        if self.ood_enabled:
            ood_score = self._compute_ood_score(class_scores_raw, valid_length)
            is_ood = ood_score >= self.ood_threshold

        # category_scores 必须先于 harmful_segments 计算（后者需用 top 类别）
        # category_scores = 异常分数 × 条件概率
        # 条件概率 P(cat_i | anomaly) = softmax(logits2)[i] / (1 - softmax(logits2)[0])
        # 最终置信度 = anomaly_score × P(cat_i | anomaly)
        category_scores = {}
        if logits2 is not None and class_scores_raw.shape[0] > 0 and valid_length > 0:
            # 取异常分数最高的 top-k 帧的类别分布平均（而非单帧 argmax），降低噪声
            k = min(3, valid_length)
            topk_indices = np.argsort(valid_scores)[-k:]
            frame_probs = class_scores_raw[topk_indices].mean(axis=0)  # [num_text]
            normal_prob = float(frame_probs[0])
            anomaly_prob = 1.0 - normal_prob

            if anomaly_prob > 1e-6:
                for idx in range(1, len(text_list)):
                    cat_key = text_list[idx]
                    cat_en = cat_key.capitalize()
                    # 条件概率 × 异常分数
                    conditional_prob = float(frame_probs[idx]) / anomaly_prob
                    category_scores[cat_en] = anomaly_score * conditional_prob
            else:
                # 视频正常，各类别分数趋近 0
                for idx in range(1, len(text_list)):
                    cat_key = text_list[idx]
                    cat_en = cat_key.capitalize()
                    category_scores[cat_en] = 0.0
        else:
            for idx in range(1, len(text_list)):
                cat_key = text_list[idx]
                cat_en = cat_key.capitalize()
                category_scores[cat_en] = 0.0

        # 改动 2: 零样本新类别扩展
        extra_category_info = {}
        if self.extra_categories and is_harmful:
            valid_features = features[:valid_length]  # [valid_length, D]
            valid_feats_norm = valid_features / (np.linalg.norm(valid_features, axis=-1, keepdims=True) + 1e-10)
            for cat_key, cat_info in self.extra_categories.items():
                text_feat = cat_info['text_feat']  # [D]
                sim = valid_feats_norm @ text_feat  # [valid_length]
                k = min(3, len(sim))
                topk_sim = float(np.sort(sim)[-k:].mean())
                cat_score = float(anomaly_score * max(0.0, topk_sim))
                if cat_score > 0:
                    cat_en = cat_key.capitalize()
                    category_scores[cat_en] = cat_score
                    extra_category_info[cat_en] = {
                        'zh': cat_info['zh'],
                        'score': cat_score,
                        'is_extra': True,
                    }

        harmful_segments = self._locate_harmful_segments(
            segment_scores, timestamps, self.threshold, fps, category_scores
        )

        keyframe_paths = self._extract_keyframes(
            video_path, harmful_segments, segment_scores, timestamps
        )

        predicted_categories = [
            cat for cat, score in sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
            if score >= self.threshold
        ]

        processing_time = time.time() - start_time
        detection_time = time.strftime('%Y-%m-%d %H:%M:%S')

        return DetectionResult(
            video_path=video_path,
            video_id=video_id,
            duration=duration,
            is_harmful=is_harmful,
            anomaly_score=anomaly_score,
            predicted_categories=predicted_categories,
            category_scores=category_scores,
            segment_scores=segment_scores,
            harmful_segments=harmful_segments,
            keyframe_paths=keyframe_paths,
            attention_weights=attention_weights,
            processing_time=processing_time,
            detection_time=detection_time,
            calibrated_score=calibrated_score,
            ood_score=ood_score,
            is_ood=is_ood,
            extra_category_info=extra_category_info,
        )

    def _compute_ood_score(self, class_scores_raw, valid_length):
        """基于类别分布的 OOD 分数。分布越平均（熵越高、max_prob 越低）越像 OOD。"""
        if valid_length == 0 or class_scores_raw.shape[0] == 0:
            return 0.5
        valid_probs = class_scores_raw[:valid_length]  # [T_valid, num_text]
        mean_probs = valid_probs.mean(axis=0)  # [num_text]
        max_prob = float(np.max(mean_probs))
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
        max_entropy = np.log(len(mean_probs))
        norm_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0
        ood_score = 0.5 * (1.0 - max_prob) + 0.5 * norm_entropy
        return float(np.clip(ood_score, 0.0, 1.0))

    def _compute_frame_quality(self, frames):
        """评估每帧质量，返回 [0,1] 质量分数数组。

        综合 Laplacian 方差（清晰度）、亮度合理性、Shannon 熵。
        """
        import cv2
        n = len(frames)
        qualities = np.ones(n, dtype=np.float32)
        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # 清晰度（Laplacian 方差）
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            clarity = min(1.0, lap_var / 500.0)
            # 亮度合理性
            brightness = float(gray.mean()) / 255.0
            bright_score = max(0.0, 1.0 - 2.0 * abs(brightness - 0.5))
            # Shannon 熵
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
            hist = hist / (hist.sum() + 1e-10)
            entropy = -np.sum(hist * np.log(hist + 1e-10))
            entropy_score = min(1.0, entropy / 8.0)
            qualities[i] = 0.5 * clarity + 0.3 * bright_score + 0.2 * entropy_score
        return qualities

    def _locate_harmful_segments(self, segment_scores, timestamps, threshold, fps, category_scores=None):
        if segment_scores.ndim > 1:
            scores_1d = segment_scores.max(axis=-1)
        else:
            scores_1d = segment_scores.copy()

        window_size = 3
        if len(scores_1d) >= window_size:
            kernel = np.ones(window_size) / window_size
            scores_1d = np.convolve(scores_1d, kernel, mode='same')

        above = scores_1d >= threshold
        regions = []
        in_region = False
        start_idx = 0
        for i in range(len(above)):
            if above[i] and not in_region:
                start_idx = i
                in_region = True
            elif not above[i] and in_region:
                regions.append((start_idx, i - 1))
                in_region = False
        if in_region:
            regions.append((start_idx, len(above) - 1))

        merged = []
        for region in regions:
            if not merged:
                merged.append(list(region))
            else:
                prev_end_idx = merged[-1][1]
                gap = timestamps[region[0]][0] - timestamps[prev_end_idx][1]
                if gap < 1.0:
                    merged[-1][1] = region[1]
                else:
                    merged.append(list(region))

        harmful_segments = []
        for start_idx, end_idx in merged:
            region_scores = scores_1d[start_idx:end_idx + 1]
            peak_idx = start_idx + int(np.argmax(region_scores))
            start_time = timestamps[start_idx][0]
            end_time = timestamps[end_idx][1]
            peak_score = float(scores_1d[peak_idx])

            # 修复 LABEL_MAP 索引错位：segment_scores 是 1D 无法取类别，
            # 改为从 category_scores 取 top 类（与 category_scores 计算保持一致）
            if category_scores:
                category_en = max(category_scores.items(), key=lambda x: x[1])[0]
            else:
                category_en = 'unknown'
            category = EN_TO_ZH.get(category_en, category_en)

            harmful_segments.append(HarmfulSegment(
                start_time=start_time,
                end_time=end_time,
                score=peak_score,
                category=category,
                category_en=category_en,
            ))

        return harmful_segments

    def _extract_keyframes(self, video_path, harmful_segments, segment_scores, timestamps):
        keyframe_paths = []
        if not harmful_segments:
            return keyframe_paths

        temp_dir = os.path.join(os.path.dirname(video_path), '.nsfw_keyframes')
        os.makedirs(temp_dir, exist_ok=True)

        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return keyframe_paths

        for seg in harmful_segments:
            best_idx = None
            best_score = -1.0
            for i, (ts_start, ts_end) in enumerate(timestamps):
                if ts_start >= seg.start_time and ts_end <= seg.end_time:
                    if segment_scores.ndim > 1:
                        score = float(segment_scores[i].max())
                    else:
                        score = float(segment_scores[i])
                    if score > best_score:
                        best_score = score
                        best_idx = i

            if best_idx is not None:
                mid_time = (timestamps[best_idx][0] + timestamps[best_idx][1]) / 2.0
                cap.set(cv2.CAP_PROP_POS_MSEC, mid_time * 1000)
                ret, frame = cap.read()
                if ret:
                    filename = f"keyframe_{os.path.splitext(os.path.basename(video_path))[0]}_{seg.start_time:.1f}_{seg.end_time:.1f}.jpg"
                    filepath = os.path.join(temp_dir, filename)
                    cv2.imwrite(filepath, frame)
                    keyframe_paths.append(filepath)

        cap.release()
        return keyframe_paths

    def _load_model(self, checkpoint_path):
        mcfg = self.config.get('model', {})
        from utils.tools import validate_clip_config
        validate_clip_config(mcfg)
        device_str = str(self.device)
        model = SVLA(
            num_class=mcfg.get('num_classes_with_normal', 8),
            embed_dim=mcfg.get('embed_dim', 512),
            visual_length=mcfg.get('visual_length', 256),
            visual_width=mcfg.get('visual_width', 512),
            visual_head=mcfg.get('visual_head', 1),
            visual_layers=mcfg.get('visual_layers', 2),
            attn_window=mcfg.get('attn_window', 8),
            prompt_prefix=mcfg.get('prompt_prefix', 10),
            prompt_postfix=mcfg.get('prompt_postfix', 10),
            device=device_str,
            shot_sim_thresh=mcfg.get('shot_sim_thresh', 0.90),
            shot_min_len=mcfg.get('shot_min_len', 12),
            shot_layers=mcfg.get('shot_layers', 1),
            shot_gamma=mcfg.get('shot_gamma', 0.05),
            pi_floor=mcfg.get('pi_floor', 0.05),
            cfa_tau=mcfg.get('cfa_tau', 0.8),
            cfa_beta=mcfg.get('cfa_beta', 0.8),
            cfa_prefix_len=mcfg.get('cfa_prefix_len', 32),
            cfa_bottleneck=mcfg.get('cfa_bottleneck', 256),
            cfa_prefix_rank=mcfg.get('cfa_prefix_rank', 16),
            cfa_dropout=mcfg.get('cfa_dropout', 0.1),
            clip_variant=mcfg.get('clip_variant', 'ViT-B/16'),
            feature_dim=mcfg.get('feature_dim', None),
        )
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
        model.to(self.device)
        return model

    def detect_batch(self, video_paths) -> List[DetectionResult]:
        results = []
        for video_path in video_paths:
            try:
                result = self.detect(video_path)
                results.append(result)
            except (FileNotFoundError, ValueError) as e:
                import warnings
                warnings.warn(f"Skipping {video_path}: {e}")
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    try:
                        original_device = self.device
                        self.device = torch.device('cpu')
                        if self.model is not None:
                            self.model.to(self.device)
                        self.feature_extractor.device = self.device
                        result = self.detect(video_path)
                        results.append(result)
                        self.device = original_device
                        if self.model is not None:
                            self.model.to(self.device)
                        self.feature_extractor.device = self.device
                    except Exception as fallback_e:
                        import warnings
                        warnings.warn(f"Failed to process {video_path} even on CPU: {fallback_e}")
                        self.device = original_device
                        if self.model is not None:
                            self.model.to(self.device)
                        self.feature_extractor.device = self.device
                else:
                    raise
        return results
