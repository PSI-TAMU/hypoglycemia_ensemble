import numpy as np
from sklearn.metrics import roc_curve

def calculate_eer_threshold(labels, preds):
    fpr, tpr, thresholds = roc_curve(labels, preds)
    fnr = 1 - tpr  # False Negative Rate
    eer_threshold = thresholds[np.nanargmin(np.abs(fpr - fnr))]
    return eer_threshold

def calculate_confusion_matrix(labels, preds, threshold):
    preds = (preds > threshold).astype(int)
    tp = np.sum((preds == 1) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    confusion_maxtrix = np.array([[tp, fp], [fn, tn]]) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return confusion_maxtrix, sensitivity, specificity