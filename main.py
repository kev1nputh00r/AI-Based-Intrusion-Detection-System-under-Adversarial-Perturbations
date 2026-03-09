from __future__ import annotations

import argparse

from .evaluate import evaluate_adversarial, evaluate_clean
from .experiments import run_ablation
from .train import save_models, train
from .utils import Paths, load_config, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid IDS training and evaluation")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_cfg = cfg["output"]

    paths = Paths(
        artifacts_dir=output_cfg["artifacts_dir"],
        reports_dir=output_cfg["reports_dir"],
        models_dir=output_cfg["models_dir"],
        metrics_file=output_cfg["metrics_file"],
    )

    detector, threshold, data = train(cfg)
    X_train, X_test, y_train, y_test, numeric_cols, _ = data

    print("Evaluating on clean test data...")
    clean_metrics = evaluate_clean(detector, X_test, y_test, threshold, cfg)

    print("Evaluating under adversarial conditions...")
    # Pass y_train so surrogate is trained on training data (no leakage)
    cfg["_y_train"] = y_train
    adv_metrics = evaluate_adversarial(
        detector, X_train, X_test, y_test, threshold, numeric_cols, cfg
    )

    ablation_results = None
    ablation_cfg = cfg.get("ablation", {})
    if ablation_cfg.get("run", False):
        print("Running threshold ablation...")
        ablation_results = run_ablation(
            detector,
            X_test,
            y_test,
            ablation_cfg.get("thresholds", [0.3, 0.5, 0.7]),
        )

    metrics = {
        "threshold": threshold,
        "clean": clean_metrics,
        "adversarial": adv_metrics,
        "ablation": ablation_results,
    }

    save_models(detector, paths)
    write_json(paths.metrics_file, metrics)

    # Save threshold separately so infer.py can load it without metrics.json
    import json, os
    threshold_path = os.path.join(paths.models_dir, 'threshold.json')
    with open(threshold_path, 'w') as f:
        json.dump({'threshold': threshold}, f)
    print(f'Threshold saved to {threshold_path}')

    print("Training and evaluation complete.")
    print(f"Metrics saved to {paths.metrics_file}")


if __name__ == "__main__":
    main()