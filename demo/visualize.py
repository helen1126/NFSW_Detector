import plotly.graph_objects as go
import base64, os
import numpy as np
from typing import List, Tuple, Optional

def create_anomaly_curve(segment_scores, timestamps, threshold=0.5):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timestamps, y=segment_scores, mode='lines', name='Anomaly Score', line=dict(color='blue')))
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold={threshold}")
    harmful_x, harmful_y = [], []
    for i, s in enumerate(segment_scores):
        if s >= threshold:
            harmful_x.append(timestamps[i])
            harmful_y.append(s)
    if harmful_x:
        fig.add_trace(go.Scatter(x=harmful_x, y=harmful_y, mode='markers', name='Harmful', marker=dict(color='red', size=6)))
    fig.update_layout(title="Anomaly Score Timeline", xaxis_title="Time (segments)", yaxis_title="Score", hovermode='x unified')
    return fig

def create_category_distribution(category_scores: dict):
    sorted_items = sorted(category_scores.items(), key=lambda x: -x[1])
    categories = [k for k, v in sorted_items]
    scores = [v for k, v in sorted_items]
    colors = ['red' if s >= 0.5 else 'green' for s in scores]
    fig = go.Figure(go.Bar(x=scores, y=categories, orientation='h', marker_color=colors))
    fig.update_layout(title="Category Distribution", xaxis_title="Score", yaxis_title="Category")
    return fig

def create_attention_heatmap(attention_weights):
    fig = go.Figure(go.Heatmap(z=attention_weights, colorscale='Viridis'))
    fig.update_layout(title="Attention Weights", xaxis_title="Segment", yaxis_title="Head")
    return fig

def create_alert_gauge(score, level):
    gauge_colors = [(0.3, 'green'), (0.5, 'yellow'), (0.8, 'orange'), (1.0, 'red')]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': f"Level: {level}"},
        gauge={'axis': {'range': [0, 1]},
               'bar': {'color': 'darkgray'},
               'steps': [{'range': [0, 0.3], 'color': 'green'},
                         {'range': [0.3, 0.5], 'color': 'yellow'},
                         {'range': [0.5, 0.8], 'color': 'orange'},
                         {'range': [0.8, 1.0], 'color': 'red'}]}
    ))
    return fig

def generate_report_html(report):
    html = f"""<html><head><style>
    body {{ font-family: Arial; padding: 20px; }}
    .header {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
    .content {{ margin-top: 15px; }}
    .harmful-item {{ border-left: 3px solid red; padding-left: 10px; margin: 10px 0; }}
    </style></head><body>
    <div class="header"><h2>Alert Report - {report.alert_level}</h2>
    <p>Video: {report.video_id} | Score: {report.anomaly_score:.4f} | Time: {report.scan_time}</p></div>
    <div class="content"><h3>Summary</h3><p>{report.summary}</p>
    <h3>Action Suggestion</h3><p>{report.action_suggestion}</p>
    <h3>Harmful Content Details</h3>"""
    for item in report.harmful_contents:
        html += f"""<div class="harmful-item"><b>{item.category_zh} ({item.category_en})</b> - Confidence: {item.confidence:.4f}<br>
        Time: {item.time_segments}<br>{item.description}</div>"""
    html += "</div></body></html>"
    return html

def prepare_keyframe_gallery(keyframe_paths, harmful_segments) -> List[Tuple[str, str]]:
    gallery = []
    for i, path in enumerate(keyframe_paths):
        if os.path.exists(path):
            caption = ""
            if i < len(harmful_segments):
                seg = harmful_segments[i]
                caption = f"{seg.start_time:.1f}s-{seg.end_time:.1f}s | {seg.category} | {seg.score:.3f}"
            gallery.append((path, caption))
    return gallery
