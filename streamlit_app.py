from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.evaluate import evaluate_adversarial, evaluate_clean
from src.infer import load_detector, load_threshold, load_new_data, compute_metrics, run_adversarial
from src.experiments import run_ablation
from src.train import save_models, train
from src.utils import Paths, load_config, write_json


def parse_csv_list(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> List[float]:
    if not value:
        return []
    values: List[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(float(item))
        except ValueError:
            continue
    return values


def parse_optional_int(value: str) -> Optional[int]:
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def build_paths(cfg: Dict[str, Any]) -> Paths:
    output_cfg = cfg["output"]
    return Paths(
        artifacts_dir=output_cfg["artifacts_dir"],
        reports_dir=output_cfg["reports_dir"],
        models_dir=output_cfg["models_dir"],
        metrics_file=output_cfg["metrics_file"],
    )


def load_metrics(metrics_path: str) -> Optional[Dict[str, Any]]:
    path = Path(metrics_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def show_report(report: Dict[str, Any]) -> None:
    if not report:
        st.write("No report available.")
        return
    report_df = pd.DataFrame(report).T
    st.dataframe(report_df, use_container_width=True)


def show_confusion_matrix(cm: List[List[int]]) -> None:
    if not cm:
        st.write("No confusion matrix available.")
        return
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
    st.dataframe(cm_df, use_container_width=True)


def show_roc_curve(roc_curve: Dict[str, Any]) -> None:
    # Intentionally left empty - matplotlib plot is used instead
    pass


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    buffer.seek(0)
    return buffer.read()


def plot_confusion_matrix(cm: List[List[int]], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], labels=["0", "1"])
    ax.set_yticks([0, 1], labels=["0", "1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="black")
    fig.tight_layout()
    return fig


def plot_roc_curve(roc_curve: Dict[str, Any], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(roc_curve["fpr"], roc_curve["tpr"], label="ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def build_summary_rows(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def _extract(report: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        attack = report.get("1", {})
        return attack.get("precision"), attack.get("recall"), attack.get("f1-score")

    clean = metrics.get("clean", {})
    if clean:
        precision, recall, f1 = _extract(clean.get("report", {}))
        rows.append(
            {
                "setting": "clean",
                "auc": clean.get("auc"),
                "accuracy": clean.get("report", {}).get("accuracy"),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    adv = metrics.get("adversarial", {})
    for name, payload in adv.items():
        precision, recall, f1 = _extract(payload.get("report", {}))
        rows.append(
            {
                "setting": name,
                "auc": payload.get("auc"),
                "accuracy": payload.get("report", {}).get("accuracy"),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return rows


st.set_page_config(page_title="Hybrid IDS Dashboard", layout="wide")

st.title("Hybrid IDS Training and Evaluation")
st.caption("Streamlit UI for training, evaluation, and adversarial analysis.")

config_path = st.sidebar.text_input("Config path", value="config.yaml")

try:
    base_cfg = load_config(config_path)
except FileNotFoundError:
    st.error("Config file not found. Update the path in the sidebar.")
    st.stop()
except Exception as exc:
    st.error(f"Failed to load config: {exc}")
    st.stop()

cfg = copy.deepcopy(base_cfg)

with st.form("settings"):
    st.subheader("Dataset")
    train_csv = st.text_input("Training CSV", value=cfg["dataset"]["train_csv"])
    test_csv = st.text_input("Testing CSV (optional)", value=cfg["dataset"].get("test_csv", ""))
    target_col = st.text_input("Target column", value=cfg["dataset"]["target_col"])
    drop_cols = st.text_input(
        "Drop columns (comma separated)",
        value=", ".join(cfg["dataset"].get("drop_cols", [])),
    )
    categorical_cols = st.text_input(
        "Categorical columns (comma separated)",
        value=", ".join(cfg["dataset"].get("categorical_cols", [])),
    )
    normalize = st.checkbox("Normalize numeric features", value=cfg["dataset"].get("normalize", True))
    test_size = st.slider("Test size", min_value=0.05, max_value=0.5, value=float(cfg["dataset"].get("test_size", 0.2)), step=0.05)
    random_state = st.number_input("Random state", min_value=0, value=int(cfg["dataset"].get("random_state", 42)))

    st.subheader("Models")
    supervised_options = cfg["model"].get("available_supervised", ["random_forest", "gradient_boosting"])
    anomaly_options = cfg["model"].get("available_anomaly", ["isolation_forest", "one_class_svm"])
    supervised = st.selectbox("Supervised model", options=supervised_options, index=supervised_options.index(cfg["model"].get("supervised", "random_forest")))
    anomaly = st.selectbox("Anomaly model", options=anomaly_options, index=anomaly_options.index(cfg["model"].get("anomaly", "isolation_forest")))
    hybrid_weight = st.slider("Hybrid weight", min_value=0.0, max_value=1.0, value=float(cfg["model"].get("hybrid_weight", 0.7)), step=0.05)
    calibrate = st.checkbox("Calibrate supervised model", value=cfg["model"].get("calibrate", True))

    st.subheader("Training")
    n_estimators = st.number_input("n_estimators", min_value=10, max_value=2000, value=int(cfg["training"].get("n_estimators", 300)), step=10)
    max_depth = st.text_input("max_depth (empty for None)", value="" if cfg["training"].get("max_depth") is None else str(cfg["training"].get("max_depth")))
    class_weight = st.text_input("class_weight", value=str(cfg["training"].get("class_weight", "balanced")))

    st.subheader("Evaluation")
    threshold_target_fpr = st.slider("Target FPR", min_value=0.0, max_value=0.2, value=float(cfg["evaluation"].get("threshold_target_fpr", 0.05)), step=0.01)
    export_confusion_matrix = st.checkbox("Export confusion matrix", value=cfg["evaluation"].get("export_confusion_matrix", True))
    export_roc_points = st.checkbox("Export ROC points", value=cfg["evaluation"].get("export_roc_points", True))

    st.subheader("Ablation")
    run_ablation_flag = st.checkbox("Run ablation", value=cfg.get("ablation", {}).get("run", True))
    ablation_thresholds = st.text_input(
        "Ablation thresholds (comma separated)",
        value=", ".join([str(x) for x in cfg.get("ablation", {}).get("thresholds", [0.3, 0.5, 0.7])]),
    )

    st.subheader("Adversarial")
    epsilon = st.slider("FGSM epsilon", min_value=0.0, max_value=0.2, value=float(cfg["adversarial"].get("epsilon", 0.05)), step=0.01)
    noise_std = st.slider("Gaussian noise std", min_value=0.0, max_value=0.2, value=float(cfg["adversarial"].get("noise_std", 0.02)), step=0.01)
    feature_mask_ratio = st.slider("Feature mask ratio", min_value=0.0, max_value=0.2, value=float(cfg["adversarial"].get("feature_mask_ratio", 0.05)), step=0.01)
    pgd_steps = st.slider("PGD steps", min_value=1, max_value=20, value=int(cfg["adversarial"].get("pgd_steps", 10)), step=1)

    submitted = st.form_submit_button("Run training and evaluation")

if submitted:
    cfg["dataset"]["train_csv"] = train_csv
    cfg["dataset"]["test_csv"] = test_csv or None
    cfg["dataset"]["target_col"] = target_col
    cfg["dataset"]["drop_cols"] = parse_csv_list(drop_cols)
    cfg["dataset"]["categorical_cols"] = parse_csv_list(categorical_cols)
    cfg["dataset"]["normalize"] = normalize
    cfg["dataset"]["test_size"] = float(test_size)
    cfg["dataset"]["random_state"] = int(random_state)

    cfg["model"]["supervised"] = supervised
    cfg["model"]["anomaly"] = anomaly
    cfg["model"]["hybrid_weight"] = float(hybrid_weight)
    cfg["model"]["calibrate"] = bool(calibrate)

    cfg["training"]["n_estimators"] = int(n_estimators)
    cfg["training"]["max_depth"] = parse_optional_int(max_depth)
    cfg["training"]["class_weight"] = class_weight

    cfg["evaluation"]["threshold_target_fpr"] = float(threshold_target_fpr)
    cfg["evaluation"]["export_confusion_matrix"] = bool(export_confusion_matrix)
    cfg["evaluation"]["export_roc_points"] = bool(export_roc_points)

    cfg["ablation"]["run"] = bool(run_ablation_flag)
    cfg["ablation"]["thresholds"] = parse_float_list(ablation_thresholds)

    cfg["adversarial"]["epsilon"] = float(epsilon)
    cfg["adversarial"]["noise_std"] = float(noise_std)
    cfg["adversarial"]["feature_mask_ratio"] = float(feature_mask_ratio)
    cfg["adversarial"]["pgd_steps"] = int(pgd_steps)

    with st.spinner("Training and evaluating..."):
        detector, threshold, data = train(cfg)
        X_train, X_test, y_train, y_test, numeric_cols, _ = data

        clean_metrics = evaluate_clean(detector, X_test, y_test, threshold, cfg)
        # Pass y_train via cfg so surrogate is trained on training data (avoids test-set leakage)
        cfg["_y_train"] = y_train
        adv_metrics = evaluate_adversarial(detector, X_train, X_test, y_test, threshold, numeric_cols, cfg)

        ablation_cfg = cfg.get("ablation", {})
        ablation_results = None
        if ablation_cfg.get("run", False):
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

        paths = build_paths(cfg)
        save_models(detector, paths)
        write_json(paths.metrics_file, metrics)

        st.session_state["metrics"] = metrics
        st.session_state["metrics_path"] = paths.metrics_file

    st.success("Run complete. Metrics and model artifacts saved.")

metrics = st.session_state.get("metrics")
metrics_path = st.session_state.get("metrics_path", cfg["output"].get("metrics_file"))

if st.button("Load latest metrics from disk"):
    loaded = load_metrics(metrics_path)
    if loaded:
        st.session_state["metrics"] = loaded
        st.success("Metrics loaded.")
    else:
        st.warning("No metrics file found.")

metrics = st.session_state.get("metrics")
if metrics:
    st.subheader("Results")
    st.write(f"Threshold: {metrics.get('threshold'):.4f}")

    st.markdown("### Summary Table")
    summary_rows = build_summary_rows(metrics)
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True)
        st.download_button(
            "Download summary CSV",
            summary_df.to_csv(index=False).encode("utf-8"),
            file_name="ids_summary.csv",
            mime="text/csv",
        )
    else:
        st.info("Summary table is not available.")

    clean = metrics.get("clean", {})
    if clean:
        st.markdown("### Clean Evaluation")
        st.metric("AUC", f"{clean.get('auc', 0.0):.4f}")
        show_report(clean.get("report", {}))
        if clean.get("confusion_matrix"):
            show_confusion_matrix(clean["confusion_matrix"])
            fig = plot_confusion_matrix(clean["confusion_matrix"], "Clean Confusion Matrix")
            st.pyplot(fig)
            st.download_button(
                "Download clean confusion matrix",
                fig_to_png_bytes(fig),
                file_name="clean_confusion_matrix.png",
                mime="image/png",
            )
            plt.close(fig)
        if clean.get("roc_curve"):
            show_roc_curve(clean["roc_curve"])
            fig = plot_roc_curve(clean["roc_curve"], "Clean ROC Curve")
            st.pyplot(fig)
            st.download_button(
                "Download clean ROC curve",
                fig_to_png_bytes(fig),
                file_name="clean_roc_curve.png",
                mime="image/png",
            )
            plt.close(fig)

    adv = metrics.get("adversarial", {})
    if adv:
        st.markdown("### Adversarial Evaluation")
        for name, payload in adv.items():
            st.markdown(f"#### {name}")
            st.metric("AUC", f"{payload.get('auc', 0.0):.4f}")
            show_report(payload.get("report", {}))
            if payload.get("confusion_matrix"):
                show_confusion_matrix(payload["confusion_matrix"])
                fig = plot_confusion_matrix(payload["confusion_matrix"], f"{name} Confusion Matrix")
                st.pyplot(fig)
                st.download_button(
                    f"Download {name} confusion matrix",
                    fig_to_png_bytes(fig),
                    file_name=f"{name}_confusion_matrix.png",
                    mime="image/png",
                )
                plt.close(fig)

            roc_curve = payload.get("roc_curve")
            if roc_curve:
                fig = plot_roc_curve(roc_curve, f"{name} ROC Curve")
                st.pyplot(fig)
                st.download_button(
                    f"Download {name} ROC curve",
                    fig_to_png_bytes(fig),
                    file_name=f"{name}_roc_curve.png",
                    mime="image/png",
                )
                plt.close(fig)

    ablation = metrics.get("ablation")
    if ablation:
        st.markdown("### Ablation")
        ablation_rows = []
        for threshold_key, payload in ablation.items():
            report = payload.get("report", {})
            attack = report.get("1", {})
            ablation_rows.append(
                {
                    "threshold": float(payload.get("threshold", threshold_key)),
                    "auc": round(float(payload.get("auc", 0)), 4),
                    "accuracy": round(float(report.get("accuracy", 0)), 4),
                    "precision (attack)": round(float(attack.get("precision", 0)), 4),
                    "recall (attack)": round(float(attack.get("recall", 0)), 4),
                    "f1 (attack)": round(float(attack.get("f1-score", 0)), 4),
                }
            )
        st.dataframe(pd.DataFrame(ablation_rows), use_container_width=True)
else:
    st.info("Run the pipeline to see results or load metrics from disk.")

# ── Inference Section ──────────────────────────────────────────────────────────
st.markdown("---")
st.header("Run on New Dataset")
st.caption("Load a previously trained model and evaluate it on any new CSV — no retraining needed.")

col1, col2 = st.columns([2, 1])
with col1:
    infer_csv = st.text_input(
        "Path to new CSV",
        placeholder="e.g. data/Friday-WorkingHours.pcap_ISCX.csv",
        key="infer_csv"
    )
with col2:
    infer_threshold = st.number_input(
        "Decision threshold (0 = use saved)",
        min_value=0.0, max_value=1.0, value=0.0, step=0.01,
        key="infer_threshold",
        help="Leave at 0 to use the threshold saved from training (recommended)"
    )

run_adv = st.checkbox("Run adversarial attacks on new dataset", value=False, key="infer_adv")

if st.button("Run Inference", type="primary"):
    if not infer_csv:
        st.warning("Please enter a CSV path above.")
    else:
        paths = build_paths(cfg)
        try:
            with st.spinner("Loading model..."):
                infer_detector = load_detector(paths.models_dir)
                saved_threshold = load_threshold(paths.models_dir)
                threshold_to_use = infer_threshold if infer_threshold > 0 else saved_threshold
                st.info(f"Using threshold: **{threshold_to_use:.4f}**")

            with st.spinner(f"Loading dataset from {infer_csv}..."):
                X_new, y_new = load_new_data(infer_csv, cfg)
                st.info(f"Loaded **{len(X_new):,}** rows — Attacks: **{int(y_new.sum()):,}** | Benign: **{int((y_new==0).sum()):,}**")

            with st.spinner("Running clean evaluation..."):
                import numpy as np
                new_scores = infer_detector.predict_proba(X_new)
                infer_clean = compute_metrics(new_scores, y_new, threshold_to_use)

            st.markdown("#### Clean Evaluation")
            st.metric("AUC", f"{infer_clean['auc']:.4f}")

            report_df = pd.DataFrame(infer_clean["report"]).T
            st.dataframe(report_df, use_container_width=True)

            cm = infer_clean["confusion_matrix"]
            fig_cm = plot_confusion_matrix(cm, "Confusion Matrix — New Dataset")
            st.pyplot(fig_cm)
            st.download_button(
                "Download confusion matrix",
                fig_to_png_bytes(fig_cm),
                file_name="infer_confusion_matrix.png",
                mime="image/png",
            )
            plt.close(fig_cm)

            if run_adv:
                with st.spinner("Running adversarial attacks (this may take a few minutes)..."):
                    infer_adv = run_adversarial(
                        infer_detector, X_new, y_new, threshold_to_use, cfg
                    )

                st.markdown("#### Adversarial Evaluation")
                adv_summary = []
                for name, payload in infer_adv.items():
                    attack = payload["report"].get("1", {})
                    adv_summary.append({
                        "setting":            name,
                        "auc":                round(payload["auc"], 4),
                        "accuracy":           round(payload["report"].get("accuracy", 0), 4),
                        "precision (attack)": round(attack.get("precision", 0), 4),
                        "recall (attack)":    round(attack.get("recall", 0), 4),
                        "f1 (attack)":        round(attack.get("f1-score", 0), 4),
                    })
                st.dataframe(pd.DataFrame(adv_summary), use_container_width=True)

                for name, payload in infer_adv.items():
                    fig_a = plot_confusion_matrix(payload["confusion_matrix"], f"{name} Confusion Matrix")
                    st.pyplot(fig_a)
                    st.download_button(
                        f"Download {name} confusion matrix",
                        fig_to_png_bytes(fig_a),
                        file_name=f"infer_{name}_confusion_matrix.png",
                        mime="image/png",
                    )
                    plt.close(fig_a)

            import json as _json
            infer_results = {
                "dataset": infer_csv,
                "threshold": threshold_to_use,
                "clean": infer_clean,
                "adversarial": infer_adv if run_adv else {},
            }
            import os
            out_path = os.path.join(
                cfg["output"].get("reports_dir", "artifacts/reports"),
                "infer_results.json"
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                _json.dump(infer_results, f, indent=2)
            st.success(f"Results saved to {out_path}")

        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Error: {e}")
            raise e




























