import os
import numpy as np

try:
    import decord
    decord_available = True
except ImportError:
    decord_available = False

try:
    import cv2
    cv2_available = True
except ImportError:
    cv2_available = False


class VideoPreprocessor:
    def __init__(self, config):
        self.supported_formats = config.get("data", {}).get("supported_formats", ["mp4", "avi", "mov", "flv", "mkv"])
        self.frame_sample_rate = config.get("data", {}).get("frame_sample_rate", 1)
        self.num_segments = config.get("data", {}).get("num_segments", 10)
        self.max_duration = config.get("inference", {}).get("max_duration", 300)

    def preprocess(self, video_path, reduced_sample_rate=False):
        self.validate_format(video_path)
        frames, fps, total_frames, duration = self._decode(video_path)
        effective_sample_rate = self.frame_sample_rate
        if reduced_sample_rate:
            effective_sample_rate = max(1, self.frame_sample_rate * 2)
        if duration > self.max_duration:
            over_count = (duration - self.max_duration) / self.max_duration
            divisor = 2 ** int(over_count)
            effective_sample_rate = max(1, effective_sample_rate // divisor)
        sampled_indices = np.arange(0, len(frames), effective_sample_rate)
        sampled_frames = frames[sampled_indices]
        uniform_indices = self.uniform_sample(sampled_frames, self.num_segments)
        final_frames = sampled_frames[uniform_indices]
        final_frame_indices = sampled_indices[uniform_indices].tolist()
        if fps > 0:
            timestamps = [(idx / fps, (idx + 1) / fps) for idx in final_frame_indices]
        else:
            timestamps = [(0.0, 0.0) for _ in final_frame_indices]
        return {
            "frames": final_frames,
            "frame_indices": final_frame_indices,
            "timestamps": timestamps,
            "fps": fps,
            "duration": duration,
            "total_frames": total_frames,
        }

    def _decode(self, video_path):
        if decord_available:
            try:
                return self._decode_decord(video_path)
            except Exception:
                pass
        if cv2_available:
            return self._decode_opencv(video_path)
        raise RuntimeError("No video decoding backend available (decord or cv2 required)")

    def _decode_decord(self, video_path):
        try:
            ctx = decord.gpu(0)
            vr = decord.VideoReader(video_path, ctx=ctx)
        except Exception:
            vr = decord.VideoReader(video_path)
        fps = float(vr.get_avg_fps())
        total_frames = len(vr)
        duration = total_frames / fps if fps > 0 else 0.0
        frames = vr[:].asnumpy()
        return frames, fps, total_frames, duration

    def _decode_opencv(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        frames_list = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_list.append(frame)
        cap.release()
        frames = np.array(frames_list, dtype=np.uint8)
        return frames, fps, total_frames, duration

    def uniform_sample(self, frames, num_frames):
        total = len(frames)
        if total <= num_frames:
            return np.arange(total)
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        return indices

    def scene_keyframes(self, frames, threshold=30.0):
        if len(frames) < 2:
            return np.arange(len(frames))
        diffs = np.mean(np.abs(frames[1:].astype(np.float32) - frames[:-1].astype(np.float32)), axis=(1, 2, 3))
        keyframe_indices = [0]
        for i, diff in enumerate(diffs):
            if diff > threshold:
                keyframe_indices.append(i + 1)
        return np.array(keyframe_indices, dtype=int)

    def validate_format(self, video_path):
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        if not os.path.isfile(video_path):
            raise ValueError(f"Path is not a file: {video_path}")
        ext = os.path.splitext(video_path)[1].lstrip(".").lower()
        if ext not in [fmt.lower() for fmt in self.supported_formats]:
            raise ValueError(f"Unsupported video format: .{ext}. Supported formats: {self.supported_formats}")
