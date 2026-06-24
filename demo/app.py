import gradio as gr
import os, yaml, time
from pipeline.inference import NSFWDetector, DetectionResult
from pipeline.alert import AlertGenerator, AlertReport
from demo.visualize import (
    create_anomaly_curve, create_category_distribution,
    create_attention_heatmap, create_alert_gauge,
    generate_report_html, prepare_keyframe_gallery
)

_detector_instance = None
_alert_generator = None

def get_detector(config_path, checkpoint_path):
    global _detector_instance, _alert_generator
    if _detector_instance is None:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        _detector_instance = NSFWDetector(config, checkpoint_path)
        _alert_generator = AlertGenerator(config)
    return _detector_instance, _alert_generator

def detect_video(video_path, config_path, checkpoint_path, progress=gr.Progress()):
    detector, alert_gen = get_detector(config_path, checkpoint_path)

    progress(0.1, desc="Loading video...")
    result = detector.detect(video_path)

    progress(0.7, desc="Generating report...")
    report = alert_gen.generate(result)

    progress(0.9, desc="Creating visualizations...")

    level_colors = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟡", "SAFE": "🟢"}
    level_emoji = level_colors.get(report.alert_level, "⚪")
    overview_md = f"## {level_emoji} Alert Level: {report.alert_level}\n"
    overview_md += f"**Anomaly Score:** {report.anomaly_score:.4f}\n\n"
    overview_md += f"**Summary:** {report.summary}\n\n"
    overview_md += f"**Action:** {report.action_suggestion}\n\n"

    category_md = "### Category Scores\n"
    for cat, score in sorted(result.category_scores.items(), key=lambda x: -x[1]):
        bar_len = int(score * 20)
        category_md += f"- **{cat}**: {'█' * bar_len}{'░' * (20 - bar_len)} {score:.4f}\n"

    timestamps = list(range(len(result.segment_scores)))
    anomaly_fig = create_anomaly_curve(result.segment_scores, timestamps, detector.threshold)
    timeline_md = f"**Duration:** {result.duration:.1f}s | **Harmful segments:** {len(result.harmful_segments)}"

    keyframe_gallery = prepare_keyframe_gallery(result.keyframe_paths, result.harmful_segments)

    report_html = generate_report_html(report)

    progress(1.0, desc="Done!")

    return (
        overview_md + "\n" + category_md,
        anomaly_fig,
        timeline_md,
        keyframe_gallery,
        report_html,
        f"Processing time: {result.processing_time:.2f}s"
    )

def create_app(config_path="configs/default.yaml", checkpoint_path=None):
    with gr.Blocks(title="NFSW Detector", theme=gr.themes.Soft()) as app:
        gr.Markdown("# NFSW Detector - 多模态有害内容审查与预警系统")
        gr.Markdown("Upload a short video to detect harmful content (smoking, blood, violence, abuse, pornography, fraud, political sensitivity)")

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Upload Video")
                config_input = gr.Textbox(value=config_path, label="Config Path")
                checkpoint_input = gr.Textbox(value=checkpoint_path or "", label="Checkpoint Path")
                detect_btn = gr.Button("Detect", variant="primary", size="lg")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Detection Overview"):
                        overview_output = gr.Markdown()
                    with gr.Tab("Timeline Analysis"):
                        anomaly_plot = gr.Plot(label="Anomaly Score Curve")
                        timeline_info = gr.Markdown()
                        video_player = gr.Video(label="Video Player")
                    with gr.Tab("Keyframes"):
                        keyframe_output = gr.Gallery(label="Keyframes", columns=3, height=400)
                    with gr.Tab("Detailed Report"):
                        report_output = gr.HTML()

        status_output = gr.Markdown()

        detect_btn.click(
            fn=detect_video,
            inputs=[video_input, config_input, checkpoint_input],
            outputs=[overview_output, anomaly_plot, timeline_info, keyframe_output, report_output, status_output]
        )

    return app

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = create_app(args.config, args.checkpoint)
    app.launch(server_port=args.port, share=args.share, max_file_size=500)
