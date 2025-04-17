import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score, f1_score

def tune_thresholds(model, X, y, thresholds=np.linspace(0, 1, 101)):
    probas = np.array(model.predict_proba(X)[:, 0])

    metrics = []
    for t in thresholds:
        preds = (probas >= t).astype(int)
        rec = recall_score(y, preds, pos_label=0)
        acc = accuracy_score(y, preds)
        f1  = f1_score(y, preds, pos_label=0)
        metrics.append((t, rec, acc, f1))

    return pd.DataFrame(metrics, columns=['Threshold', 'Recall', 'Accuracy', 'F1'])