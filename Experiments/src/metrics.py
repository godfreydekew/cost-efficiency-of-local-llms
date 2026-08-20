import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report


def compute_metrics(eval_pred):
    """
    Computes accuracy and macro F1 score.
    Used as callback in HuggingFace Trainer.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': round(float(accuracy_score(labels, preds)), 4),
        'macro_f1': round(float(f1_score(labels, preds, average='macro', zero_division=0)), 4),
    }


def evaluate_predictions(labels, preds, label_names_dict=None):
    """
    Computes accuracy, macro F1, and per-class metrics.
    """
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    
    per_class_f1 = {}
    if label_names_dict:
        valid_label_keys = sorted(list(label_names_dict.keys()))
        raw_per_class = f1_score(labels, preds, labels=valid_label_keys, average=None, zero_division=0)
        for idx, key in enumerate(valid_label_keys):
            label_str = label_names_dict[key]
            per_class_f1[label_str] = round(float(raw_per_class[idx]), 4)

    return {
        'accuracy': round(float(acc), 4),
        'macro_f1': round(float(macro_f1), 4),
        'per_class_f1': per_class_f1
    }
