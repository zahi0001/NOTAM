# =============================================================================
# evaluate.py — Shared evaluation utilities for the NOTAM ML pipeline
#
# Used by both 03_model_training.ipynb and any future evaluation scripts.
# Centralising these here means plots are consistent across all notebooks.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    accuracy_score,
)
from pathlib import Path

from ml.config import CRITICALITY_LEVELS

CRIT_COLORS = {
    'HIGH':   '#d62728',
    'MEDIUM': '#ff7f0e',
    'LOW':    '#1f77b4',
    'INFO':   '#2ca02c',
}


# ---------------------------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_name: str,
    y_true,
    y_pred,
    save_path: Path = None,
) -> dict:
    """
    Prints a classification report and plots a confusion matrix.
    Returns a metrics dict for comparison tables.

    Args:
        model_name: Display name for the model
        y_true:     Ground truth labels
        y_pred:     Predicted labels
        save_path:  Optional path to save the confusion matrix PNG

    Returns:
        Dict with Accuracy, Macro F1, Weighted F1, HIGH F1
    """
    print(f'\n{"-" * 60}')
    print(f'  {model_name}')
    print(f'{"-" * 60}')

    print(classification_report(
        y_true, y_pred,
        labels=CRITICALITY_LEVELS,
        target_names=CRITICALITY_LEVELS,
        zero_division=0,
    ))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=CRITICALITY_LEVELS)
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay(cm, display_labels=CRITICALITY_LEVELS).plot(
        ax=ax, colorbar=False, cmap='Blues'
    )
    ax.set_title(f'{model_name} — Confusion Matrix', fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    acc      = accuracy_score(y_true, y_pred)
    macro    = f1_score(y_true, y_pred, average='macro',    labels=CRITICALITY_LEVELS, zero_division=0)
    weighted = f1_score(y_true, y_pred, average='weighted', labels=CRITICALITY_LEVELS, zero_division=0)
    high_f1  = f1_score(y_true, y_pred, average=None,       labels=CRITICALITY_LEVELS, zero_division=0)[
        CRITICALITY_LEVELS.index('HIGH')
    ]

    return {
        'Model':       model_name,
        'Accuracy':    round(acc * 100, 1),
        'Macro F1':    round(macro * 100, 1),
        'Weighted F1': round(weighted * 100, 1),
        'HIGH F1':     round(high_f1 * 100, 1),
    }


# ---------------------------------------------------------------------------
# Model comparison plot
# ---------------------------------------------------------------------------

def plot_comparison(metrics_list: list[dict], save_path: Path = None):
    """
    Side-by-side bar chart comparing multiple models across all metrics.

    Args:
        metrics_list: List of dicts returned by evaluate_model()
        save_path:    Optional path to save the PNG
    """
    import pandas as pd
    comparison = pd.DataFrame(metrics_list).set_index('Model')

    fig, ax = plt.subplots(figsize=(10, 5))
    x     = np.arange(len(comparison.columns))
    width = 0.8 / len(comparison)
    colors = ['#4878cf', '#6acc65', '#d65f5f', '#b47cc7']

    for i, (model_name, row) in enumerate(comparison.iterrows()):
        offset = (i - len(comparison) / 2 + 0.5) * width
        bars = ax.bar(x + offset, row.values, width * 0.9,
                      label=model_name, color=colors[i % len(colors)],
                      alpha=0.85, edgecolor='white')
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f'{bar.get_height():.1f}',
                ha='center', va='bottom', fontsize=8
            )

    ax.set_xticks(x)
    ax.set_xticklabels(comparison.columns)
    ax.set_ylabel('Score (%)')
    ax.set_ylim(0, 110)
    ax.set_title('Model Comparison — Test Set Metrics', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    print('\nWinner per metric:')
    for col in comparison.columns:
        winner = comparison[col].idxmax()
        delta  = comparison[col].max() - comparison[col].min()
        print(f'  {col:<14} -> {winner} (+{delta:.1f}%)')


# ---------------------------------------------------------------------------
# UMAP / t-SNE visualization
# ---------------------------------------------------------------------------

def plot_embeddings_2d(
    coords: np.ndarray,
    labels,
    anomaly_mask=None,
    method: str = 'UMAP',
    save_path: Path = "notebooks/figures",
):
    """
    Plots 2D embedding coordinates colored by criticality label.
    Optionally overlays anomaly flags as a second panel.

    Args:
        coords:       (n, 2) array of 2D coordinates
        labels:       Criticality label per point
        anomaly_mask: Optional boolean array — True = anomalous
        method:       'UMAP' or 't-SNE' (used in title only)
        save_path:    Optional path to save the PNG
    """
    n_panels = 2 if anomaly_mask is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(12 * n_panels / 2 + 6, 7))
    if n_panels == 1:
        axes = [axes]

    # Panel 1: colored by criticality
    for crit in CRITICALITY_LEVELS:
        mask = np.array(labels) == crit
        axes[0].scatter(
            coords[mask, 0], coords[mask, 1],
            c=CRIT_COLORS[crit],
            label=f'{crit} (n={mask.sum()})',
            alpha=0.5, s=10, edgecolors='none'
        )
    axes[0].set_title(f'{method} — Colored by Criticality', fontweight='bold')
    axes[0].legend(markerscale=2.5, title='Criticality')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Panel 2: anomalies highlighted
    if anomaly_mask is not None:
        axes[1].scatter(
            coords[~anomaly_mask, 0], coords[~anomaly_mask, 1],
            c='#bbbbbb', alpha=0.3, s=8, edgecolors='none', label='Normal'
        )
        axes[1].scatter(
            coords[anomaly_mask, 0], coords[anomaly_mask, 1],
            c='#d62728', alpha=0.85, s=30,
            edgecolors='white', linewidths=0.4,
            label=f'Anomaly (n={anomaly_mask.sum()})', zorder=5
        )
        axes[1].set_title(f'{method} — Anomalies Highlighted', fontweight='bold')
        axes[1].legend(markerscale=2)
        axes[1].set_xticks([])
        axes[1].set_yticks([])

    plt.suptitle(f'NOTAM Sentence Embedding Space ({method})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
