import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay, confusion_matrix

def plot_roc_curve(y_true_pos, y_score):
    fpr, tpr, _ = roc_curve(y_true_pos, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig

def plot_pr_curve(y_true_pos, y_score):
    precision, recall, _ = precision_recall_curve(y_true_pos, y_score)
    # PR AUC (AP) se puede aproximar con el trapecio o calcular fuera; aquí sólo graficamos
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    fig.tight_layout()
    return fig

def plot_confusion(y_true, y_pred, labels_order=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_order if labels_order else None)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Matriz de confusión")
    fig.tight_layout()
    return fig
