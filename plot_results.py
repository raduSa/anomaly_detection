import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
    precision_recall_curve, f1_score, fbeta_score, balanced_accuracy_score,
    confusion_matrix,
)

default_beta = 2.0

def sweep_threshold(y_test, scores, beta: float = default_beta):
    # beta -> weighs recall more than precision - good for fraud.

    precisions, recalls, thresholds = precision_recall_curve(y_test, scores)

    # F-beta at each threshold
    beta2 = beta ** 2
    fbetas = (1 + beta2) * precisions[:-1] * recalls[:-1] / (
        beta2 * precisions[:-1] + recalls[:-1] + 1e-9
    )
    best_idx = int(np.argmax(fbetas))
    best_threshold = thresholds[best_idx]
    best_fbeta = fbetas[best_idx]

    print(f'\nThreshold sweep (F{beta} maximisation)')
    print(f'Best threshold : {best_threshold:.4f}')
    print(f'F{beta} at threshold: {best_fbeta:.4f}')
    print(f'Precision : {precisions[best_idx]:.4f}')
    print(f'Recall : {recalls[best_idx]:.4f}')

    return best_threshold


def evaluate(y_test, scores, threshold: float, model_name: str, out_path: str = None):
    y_pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    output_lines = [
        f'\n{model_name} eval',
        classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'], digits=4),
        f'ROC-AUC          : {roc_auc_score(y_test, scores):.4f}',
        f'Avg Precision    : {average_precision_score(y_test, scores):.4f}',
        f'Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}',
        f'F1 (fraud)       : {f1_score(y_test, y_pred):.4f}',
        f'TP={tp}  TN={tn}  FP={fp}  FN={fn}',
    ]
    report = '\n'.join(output_lines)
    print(report)

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            f.write(report + '\n')
        print(f'Report saved to {out_path}')

    return y_pred


def evaluate_per_class(attack_types, scores, skip_labels=None, out_path=None):
    skip = set(skip_labels) if skip_labels else set()
    lines = ['Per-class Average Precision:']
    for cls in sorted(np.unique(attack_types)):
        if cls == 'BENIGN' or cls in skip:
            continue
        mask = (attack_types == cls) | (attack_types == 'BENIGN')
        ap = average_precision_score(
            (attack_types[mask] != 'BENIGN').astype(int), scores[mask]
        )
        lines.append(f'  {cls:<50}  AP={ap:.4f}  n={(attack_types == cls).sum()}')
    report = '\n'.join(lines)
    print(f'\n{report}')
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            f.write(report + '\n')
        print(f'Per-class AP saved to {out_path}')


def plot_results(y_test, y_pred, scores, model_name: str, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'{model_name} — Evaluation', fontsize=13, fontweight='bold')

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=['Legit', 'Fraud'],
        ax=axes[0], colorbar=False,
    )
    axes[0].set_title('Confusion Matrix')

    RocCurveDisplay.from_predictions(y_test, scores, ax=axes[1], name=model_name)
    axes[1].plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    axes[1].set_title('ROC Curve')

    PrecisionRecallDisplay.from_predictions(y_test, scores, ax=axes[2], name=model_name)
    axes[2].set_title('Precision-Recall Curve')

    plt.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.show()
    
    print(f'Plot saved to {out_path}')


def plot_loss_curves(train_losses, val_losses, title='', out_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='train')
    ax.plot(epochs, val_losses, label='val', linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title(f'{title} — Training Curves')
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Loss curve saved to {out_path}')
