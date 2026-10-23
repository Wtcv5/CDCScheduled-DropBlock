import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, f1_score

def ordinal_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    abs_err = np.abs(y_pred - y_true)
    return {'accuracy': float(accuracy_score(y_true, y_pred)), 'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)), 'macro_precision': float(p), 'macro_recall': float(r), 'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)), 'rank_mae': float(abs_err.mean()), 'cross_grade_error_rate_ge2': float(np.mean(abs_err >= 2))}
