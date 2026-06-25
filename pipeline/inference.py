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


class NSFWDetector:
    def __init__(self, config, checkpoint_path=None):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.config = config
        self.preprocessor = VideoPreprocessor(config)
        clip_variant = config.get('model', {}).get('clip_variant', 'ViT-B/16')
        self.feature_extractor = CLIPFeatureExtractor(clip_variant, self.device)
        self.model = None
        if checkpoint_path is not None:
            self.model = self._load_model(checkpoint_path)
            self.model.eval()
        self.threshold = config['inference']['anomaly_threshold']

    def detect(self, video_path) -> DetectionResult:
        start_time = time.time()
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        ext = os.path.splitext(video_path)[1].lower()
        supported_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        if ext not in supported_formats:
            raise ValueError(f"Unsupported video format: {ext}")

        try:
            preprocess_result = self.preprocessor.preprocess(video_path)
        except Exception:
            preprocess_result = self.preprocessor.preprocess(video_path, reduced_sample_rate=True)
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

        # 处理特征到模型期望的长度（visual_length），与训练时的 process_feat 一致
        visual_length = self.model.visual_length
        features, valid_length = process_feat(features, visual_length)

        with torch.inference_mode():
            features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
            features_tensor = features_tensor.unsqueeze(0)
            B, S, D = features_tensor.shape
            lengths = torch.tensor([valid_length], dtype=torch.long)
            padding_mask = torch.arange(S, device=self.device).unsqueeze(0) >= lengths.unsqueeze(1).to(self.device)
            text_list = get_prompt_text(self.config.get('label_map', {}))
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

        anomaly_score = float(np.max(segment_scores.reshape(-1)))

        is_harmful = anomaly_score >= self.threshold

        harmful_segments = self._locate_harmful_segments(
            segment_scores, timestamps, self.threshold, fps
        )

        keyframe_paths = self._extract_keyframes(
            video_path, harmful_segments, segment_scores, timestamps
        )

        # category_scores = 异常分数 × 条件概率
        # 条件概率 P(cat_i | anomaly) = softmax(logits2)[i] / (1 - softmax(logits2)[0])
        # 最终置信度 = anomaly_score × P(cat_i | anomaly)
        category_scores = {}
        if logits2 is not None and class_scores_raw.shape[0] > 0:
            # 取异常分数最高的帧的类别分布（而非所有帧的最大值）
            anomaly_frame_idx = int(np.argmax(segment_scores.reshape(-1)))
            frame_probs = class_scores_raw[anomaly_frame_idx]  # [num_text]
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
        )

    def _locate_harmful_segments(self, segment_scores, timestamps, threshold, fps):
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

            if segment_scores.ndim > 1:
                class_preds = segment_scores[peak_idx]
                primary_cat_idx = int(np.argmax(class_preds))
            else:
                primary_cat_idx = 0

            category_en = LABEL_MAP.get(primary_cat_idx, {}).get('en', 'unknown')
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
