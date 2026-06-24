import os
import subprocess
import numpy as np

try:
    import decord
except ImportError:
    decord = None

try:
    import cv2
except ImportError:
    cv2 = None


def get_video_info(video_path) -> dict:
    if decord is not None:
        try:
            vr = decord.VideoReader(video_path)
            fps = float(vr.get_avg_fps())
            total_frames = len(vr)
            duration = total_frames / fps if fps > 0 else 0.0
            frame = vr[0].asnumpy()
            height, width = frame.shape[:2]
            codec = ""
            return {
                "width": width,
                "height": height,
                "fps": fps,
                "duration": duration,
                "total_frames": total_frames,
                "codec": codec,
            }
        except Exception:
            pass

    if cv2 is not None:
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((codec_int >> 8 * i) & 0xFF) for i in range(4)])
        cap.release()
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "duration": duration,
            "total_frames": total_frames,
            "codec": codec,
        }

    return {
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "duration": 0.0,
        "total_frames": 0,
        "codec": "",
    }


def convert_format(input_path, output_path, output_format="mp4"):
    subprocess.run(
        ["ffmpeg", "-i", input_path, output_path],
        check=True,
        capture_output=True,
    )


def validate_video(video_path) -> bool:
    if not os.path.isfile(video_path):
        return False
    if os.path.getsize(video_path) <= 0:
        return False

    if decord is not None:
        try:
            vr = decord.VideoReader(video_path)
            _ = len(vr)
            return True
        except Exception:
            pass

    if cv2 is not None:
        try:
            cap = cv2.VideoCapture(video_path)
            opened = cap.isOpened()
            cap.release()
            return opened
        except Exception:
            pass

    return False


def extract_frame_at_timestamp(video_path, timestamp) -> np.ndarray:
    if decord is not None:
        try:
            vr = decord.VideoReader(video_path)
            fps = float(vr.get_avg_fps())
            frame_idx = int(timestamp * fps)
            frame_idx = max(0, min(frame_idx, len(vr) - 1))
            frame = vr[frame_idx].asnumpy()
            return frame
        except Exception:
            pass

    if cv2 is not None:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame

    return np.array([])


def save_frame_as_jpeg(frame, output_path):
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, bgr_frame)


def extract_frames_by_interval(video_path, interval_sec=1.0) -> list:
    results = []
    info = get_video_info(video_path)
    duration = info["duration"]
    if duration <= 0 or interval_sec <= 0:
        return results

    current_time = 0.0
    while current_time < duration:
        frame = extract_frame_at_timestamp(video_path, current_time)
        if frame.size > 0:
            results.append((frame, current_time))
        current_time += interval_sec

    return results


def extract_clip(video_path, start_time, end_time, output_path):
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            video_path,
            "-ss",
            str(start_time),
            "-to",
            str(end_time),
            "-c",
            "copy",
            output_path,
        ],
        check=True,
        capture_output=True,
    )


def extract_thumbnails(video_path, num_thumbnails=5, output_dir=".") -> list:
    info = get_video_info(video_path)
    duration = info["duration"]
    if duration <= 0 or num_thumbnails <= 0:
        return []

    os.makedirs(output_dir, exist_ok=True)
    output_paths = []

    for i in range(num_thumbnails):
        timestamp = duration * (i + 1) / (num_thumbnails + 1)
        frame = extract_frame_at_timestamp(video_path, timestamp)
        if frame.size > 0:
            filename = f"thumbnail_{i + 1}.jpg"
            filepath = os.path.join(output_dir, filename)
            save_frame_as_jpeg(frame, filepath)
            output_paths.append(filepath)

    return output_paths
