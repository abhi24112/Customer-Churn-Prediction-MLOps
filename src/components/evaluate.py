import logging
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Styling
plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 11,
})
PALETTE = ['#2196F3', '#F44336']
sns.set_palette(PALETTE)

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def evaluate_model(
    model_name: str,
    run_name: str,
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    threshold: float = 0.5,
    optimize_threshold: bool = False,
):
    
    try:
        logging.info("Model Evaluation is Started...")

        y_prob = model.predict_proba(x_test)[:, 1]

        chosen_threshold = float(threshold)
        if optimize_threshold:
            precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_prob)

            if len(thresholds) > 0:
                f1_curve = (2 * precision_curve[:-1] * recall_curve[:-1]) / (
                    precision_curve[:-1] + recall_curve[:-1] + 1e-12
                )
                best_idx = int(f1_curve.argmax())
                chosen_threshold = float(thresholds[best_idx])

        # Predictions (thresholded)
        y_pred = (y_prob >= chosen_threshold).astype(int)

        # Metrics
        precision_metric = precision_score(y_test, y_pred, zero_division=0)
        recall_metric = recall_score(y_test, y_pred, zero_division=0)
        accuracy_metric = accuracy_score(y_test, y_pred)
        f1_metric = f1_score(y_test, y_pred, zero_division=0)
        pr_auc = average_precision_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        # Create plots directory with timestamp to avoid overwrites
        plot_dir = ".logs/model_plots"
        os.makedirs(plot_dir, exist_ok=True)

        # --- Confusion Matrix Heatmap ---
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, cmap="Blues", ax=ax)
        ax.set_title(f"{model_name} Confusion Matrix")
        cm_path = os.path.join(plot_dir, f"confusion_matrix_{run_name}_{model_name}.png")
        fig_cm.savefig(cm_path, format="png")
        plt.close(fig_cm)

        # --- Precision-Recall Curve ---
        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
        fig_pr, ax = plt.subplots()
        ax.plot(recall, precision)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{model_name} Precision-Recall Curve (AP={pr_auc:.3f})")
        pr_path = os.path.join(plot_dir, f"precision_recall_curve_{run_name}_{model_name}.png")
        fig_pr.savefig(pr_path, format="png")
        plt.close(fig_pr)

        logging.info("Evaluation of model is completed successfully")

        metrics = {
            "accuracy": float(accuracy_metric),
            "precision": float(precision_metric),
            "recall": float(recall_metric),
            "f1": float(f1_metric),
            "pr_auc": float(pr_auc),
            "threshold": float(chosen_threshold),
        }

        artifacts = {
            f"confusion_matrix_{model_name}.png": cm_path,
            f"precision_recall_curve_{model_name}.png": pr_path,
        }

        return metrics, artifacts
    
    except Exception as e:
        logging.error("Error while evaluating the model")
        raise ValueError(f"Error while evaluating the model: {e}")