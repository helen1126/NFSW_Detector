import argparse
import os
import yaml
import torch
import numpy as np
import random

from models.svla import SVLA
from engine.train import Trainer
from engine.evaluate import Evaluator
from pipeline.inference import NSFWDetector
from pipeline.alert import AlertGenerator
from data.dataset import create_dataloaders
from utils.logger import setup_logger
from utils.tools import validate_clip_config

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║           NFSW Detector - 多模态有害内容审查与预警系统          ║
║           Shot-Conditioned Vision-Language Adaptation         ║
╚══════════════════════════════════════════════════════════════╝
"""


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def print_config_summary(config):
    print(f"Model: {config['model']['name']}")
    print(f"CLIP: {config['model']['clip_variant']}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    print(f"Classes: {config['model']['num_classes']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch Size: {config['training']['batch_size']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NFSW Detector CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--config", default="configs/default.yaml")
    train_parser.add_argument("--resume", default=None)
    train_parser.add_argument("--device", default=None)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    evaluate_parser.add_argument("--config", default="configs/default.yaml")
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--output", default="results/evaluation")
    evaluate_parser.add_argument("--device", default=None)

    detect_parser = subparsers.add_parser("detect", help="Detect NSFW content in video")
    detect_parser.add_argument("--config", default="configs/default.yaml")
    detect_parser.add_argument("--checkpoint", required=True)
    detect_parser.add_argument("--video", required=True)
    detect_parser.add_argument("--output", default="results/detection")
    detect_parser.add_argument("--threshold", type=float, default=None)
    detect_parser.add_argument("--save-frames", action="store_true")
    detect_parser.add_argument("--num-segments", type=int, default=None,
                              help="采样帧数，覆盖配置默认值")

    demo_parser = subparsers.add_parser("demo", help="Launch demo application")
    demo_parser.add_argument("--config", default="configs/default.yaml")
    demo_parser.add_argument("--checkpoint", default=None)
    demo_parser.add_argument("--port", type=int, default=7860)
    demo_parser.add_argument("--share", action="store_true")

    export_parser = subparsers.add_parser("export", help="Export model to ONNX or TorchScript")
    export_parser.add_argument("--checkpoint", required=True)
    export_parser.add_argument("--format", choices=["onnx", "torchscript"], default="onnx")
    export_parser.add_argument("--output", default="exports/model")
    export_parser.add_argument("--config", default="configs/default.yaml")

    serve_parser = subparsers.add_parser("serve", help="Launch FastAPI RESTful API server")
    serve_parser.add_argument("--config", default="configs/default.yaml")
    serve_parser.add_argument("--checkpoint", default=None, help="模型权重路径，不传则 API 以无模型模式启动")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true", help="开发模式热重载")

    args = parser.parse_args()

    if args.command == "train":
        config = load_config(args.config)
        print(BANNER)
        print_config_summary(config)
        seed = config.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if "cuda" in device and not torch.cuda.is_available():
            print(f"Warning: CUDA not available, falling back to CPU")
            device = "cpu"
        mcfg = config["model"]
        validate_clip_config(mcfg)
        model = SVLA(
            num_class=mcfg["num_classes_with_normal"], embed_dim=mcfg["embed_dim"],
            visual_length=mcfg["visual_length"], visual_width=mcfg["visual_width"],
            visual_head=mcfg["visual_head"], visual_layers=mcfg["visual_layers"],
            attn_window=mcfg["attn_window"], prompt_prefix=mcfg["prompt_prefix"],
            prompt_postfix=mcfg["prompt_postfix"], device=device,
            shot_sim_thresh=mcfg["shot_sim_thresh"], shot_min_len=mcfg["shot_min_len"],
            shot_layers=mcfg["shot_layers"], shot_gamma=mcfg["shot_gamma"],
            pi_floor=mcfg["pi_floor"], cfa_tau=mcfg["cfa_tau"], cfa_beta=mcfg["cfa_beta"],
            cfa_prefix_len=mcfg["cfa_prefix_len"], cfa_bottleneck=mcfg["cfa_bottleneck"],
            cfa_prefix_rank=mcfg["cfa_prefix_rank"], cfa_dropout=mcfg["cfa_dropout"],
            clip_variant=mcfg.get("clip_variant", "ViT-B/16"),
            feature_dim=mcfg.get("feature_dim", None),
        )
        normal_loader, anomaly_loader, test_loader = create_dataloaders(config)
        label_map = config.get("label_map", {})
        gt_path = config.get("data", {}).get("gt_path", "list/gt_Sva_500_abnormal_is_1.npy")
        gt = np.load(gt_path) if os.path.exists(gt_path) else np.array([])
        trainer = Trainer(config, model, label_map, device=device)
        if args.resume:
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        best_metrics = trainer.train(normal_loader, anomaly_loader, test_loader, gt)
        print(f"Best metrics: {best_metrics}")

    elif args.command == "evaluate":
        config = load_config(args.config)
        print(BANNER)
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if "cuda" in device and not torch.cuda.is_available():
            print(f"Warning: CUDA not available, falling back to CPU")
            device = "cpu"
        mcfg = config["model"]
        validate_clip_config(mcfg)
        model = SVLA(
            num_class=mcfg["num_classes_with_normal"], embed_dim=mcfg["embed_dim"],
            visual_length=mcfg["visual_length"], visual_width=mcfg["visual_width"],
            visual_head=mcfg["visual_head"], visual_layers=mcfg["visual_layers"],
            attn_window=mcfg["attn_window"], prompt_prefix=mcfg["prompt_prefix"],
            prompt_postfix=mcfg["prompt_postfix"], device=device,
            shot_sim_thresh=mcfg["shot_sim_thresh"], shot_min_len=mcfg["shot_min_len"],
            shot_layers=mcfg["shot_layers"], shot_gamma=mcfg["shot_gamma"],
            pi_floor=mcfg["pi_floor"], cfa_tau=mcfg["cfa_tau"], cfa_beta=mcfg["cfa_beta"],
            cfa_prefix_len=mcfg["cfa_prefix_len"], cfa_bottleneck=mcfg["cfa_bottleneck"],
            cfa_prefix_rank=mcfg["cfa_prefix_rank"], cfa_dropout=mcfg["cfa_dropout"],
            clip_variant=mcfg.get("clip_variant", "ViT-B/16"),
            feature_dim=mcfg.get("feature_dim", None),
        )
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        label_map = config.get("label_map", {})
        evaluator = Evaluator(config, model, label_map, device=device)
        gt_path = config.get("data", {}).get("gt_path", "list/gt_Sva_500_abnormal_is_1.npy")
        gt = np.load(gt_path) if os.path.exists(gt_path) else np.array([])
        _, _, test_loader = create_dataloaders(config)
        results = evaluator.evaluate(test_loader, gt)
        evaluator.export_results(results, args.output)

    elif args.command == "detect":
        config = load_config(args.config)
        print(BANNER)
        if args.threshold is not None:
            config["inference"]["anomaly_threshold"] = args.threshold
        detector = NSFWDetector(config, checkpoint_path=args.checkpoint)
        result = detector.detect(args.video, num_segments=args.num_segments)
        alert_generator = AlertGenerator(config)
        report = alert_generator.generate(result)
        os.makedirs(args.output, exist_ok=True)
        alert_generator.export_json(report, os.path.join(args.output, "report.json"))
        print(f"Anomaly Score: {result.anomaly_score:.4f} | Harmful: {result.is_harmful} | Categories: {result.predicted_categories}")

    elif args.command == "demo":
        from demo.app import create_app
        app = create_app(config_path=args.config, checkpoint_path=args.checkpoint)
        app.launch(server_port=args.port, share=args.share, max_file_size=500 * 1024 * 1024)

    elif args.command == "export":
        config = load_config(args.config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if "cuda" in device and not torch.cuda.is_available():
            print(f"Warning: CUDA not available, falling back to CPU")
            device = "cpu"
        mcfg = config["model"]
        validate_clip_config(mcfg)
        model = SVLA(
            num_class=mcfg["num_classes_with_normal"], embed_dim=mcfg["embed_dim"],
            visual_length=mcfg["visual_length"], visual_width=mcfg["visual_width"],
            visual_head=mcfg["visual_head"], visual_layers=mcfg["visual_layers"],
            attn_window=mcfg["attn_window"], prompt_prefix=mcfg["prompt_prefix"],
            prompt_postfix=mcfg["prompt_postfix"], device=device,
            shot_sim_thresh=mcfg["shot_sim_thresh"], shot_min_len=mcfg["shot_min_len"],
            shot_layers=mcfg["shot_layers"], shot_gamma=mcfg["shot_gamma"],
            pi_floor=mcfg["pi_floor"], cfa_tau=mcfg["cfa_tau"], cfa_beta=mcfg["cfa_beta"],
            cfa_prefix_len=mcfg["cfa_prefix_len"], cfa_bottleneck=mcfg["cfa_bottleneck"],
            cfa_prefix_rank=mcfg["cfa_prefix_rank"], cfa_dropout=mcfg["cfa_dropout"],
            clip_variant=mcfg.get("clip_variant", "ViT-B/16"),
            feature_dim=mcfg.get("feature_dim", None),
        )
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        os.makedirs(args.output, exist_ok=True)
        if args.format == "onnx":
            dummy_input = torch.randn(1, mcfg["visual_length"], mcfg["embed_dim"])
            onnx_path = os.path.join(args.output, "model.onnx")
            torch.onnx.export(
                model, dummy_input, onnx_path,
                input_names=["input"], output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )
            print(f"Model exported to ONNX: {onnx_path}")
        elif args.format == "torchscript":
            dummy_input = torch.randn(1, mcfg["visual_length"], mcfg["embed_dim"])
            ts_path = os.path.join(args.output, "model.pt")
            scripted = torch.jit.trace(model, dummy_input)
            scripted.save(ts_path)
            print(f"Model exported to TorchScript: {ts_path}")

    elif args.command == "serve":
        print(BANNER)
        from api.app import init_state
        import uvicorn
        init_state(args.config, args.checkpoint)
        loaded = "已加载" if args.checkpoint else "未加载（无模型模式，仅元数据接口可用）"
        print(f"FastAPI 服务启动中... 模型: {loaded}")
        print(f"Swagger UI:  http://{args.host}:{args.port}/docs")
        print(f"ReDoc:        http://{args.host}:{args.port}/redoc")
        uvicorn.run(
            "api.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )

    else:
        parser.print_help()
