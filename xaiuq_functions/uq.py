from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, brier_score_loss, log_loss
import pandas as pd
from IPython.display import display

def compute_metrics(metrics_list, y_true):
    """
    Compute and display classification and calibration metrics for multiple models.

    Parameters:
        metrics_list (list of tuples): Each tuple is (model_name, y_pred, y_prob)
            - model_name (str): label for the model
            - y_pred (array-like): predicted class labels
            - y_prob (array-like): predicted probabilities for the positive class
        y_true (array-like): true class labels

    Returns:
        pandas.DataFrame: one row per model, columns are Accuracy, F1-score, Recall,
                          Precision, ROC AUC, Brier Score, Log loss
    """
    rows = []
    for name, y_pred, y_prob in metrics_list:
        rows.append({
            'Model':        name,
            'Accuracy':     accuracy_score(y_true, y_pred),
            'F1-score':     f1_score(y_true, y_pred),
            'Recall':       recall_score(y_true, y_pred),
            'Precision':    precision_score(y_true, y_pred),
            'ROC AUC':      roc_auc_score(y_true, y_prob),
            'Brier Score':  brier_score_loss(y_true, y_prob),
            'Log loss':     log_loss(y_true, y_prob),
        })
    df = pd.DataFrame(rows).set_index('Model').round(3)
    return df


import seaborn as sns
import matplotlib.pyplot as plt

def conformal_probabilities(threshold, h_prob_df, highlight_cases=None, epsilon=0.15, alpha=0.1):
    """
    Plot conformal intervals wider than epsilon and highlight selected cases.

    Parameters
    ----------
    threshold : float
    h_prob_df : pd.DataFrame
    highlight_cases : list of (pd.Series, str), optional
        Each tuple is (row, label) to mark on the plot.
    epsilon : float
        Minimum interval width to include in the plot.
    """
    if highlight_cases is None:
        highlight_cases = []

    df = (
        h_prob_df
        .assign(orig_idx=h_prob_df.index)
        .sort_values('p_mid')
        .reset_index(drop=True)
    )
    df2 = df[df['interval_width'] > epsilon].reset_index(drop=True)
    idx_thr2 = df2.index[df2['p_ivap'] >= threshold][0]

    sns.set_style('whitegrid')
    palette = sns.color_palette("viridis", 30)
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.scatter(df2.index, df2['h_prob'],      s=120, color=palette[1],  alpha=0.7, label='no calibrado')
    ax.scatter(df2.index, df2['p1'],          s=90,  color=palette[20], alpha=0.8, label='p1 superior')
    ax.scatter(df2.index, df2['p0'],          s=90,  color=palette[25], alpha=0.8, label='p0 inferior')
    ax.scatter(df2.index, df2['p_ivap'],       s=50,  color=palette[10], alpha=0.9, label='p unificado')
    ax.scatter(df2.index, df2['QoL'],         s=30,  color=palette[7],  alpha=0.9, label='valor real')
    ax.step(   df2.index, df2['interval_width'], where='mid', lw=1.5, color=palette[25], label='intervalo')

    ax.axhline(epsilon, color='red', linestyle='--', label=f'ε = {epsilon}')
    ax.axhline(alpha,   color='orange', linestyle='-.', label=f'α = {alpha}')
    ax.axhline(threshold, color='red', linewidth=1.5, alpha=0.5,      label=f'threshold = {threshold:.2f}')
    ax.axvline(idx_thr2,   color='black', linewidth=1.5, linestyle=':', alpha=0.5, label=f'thr idx = {idx_thr2}')

    for row, lbl in highlight_cases:
        pos = df2.index[df2['orig_idx'] == row.name]
        if not pos.empty:
            i = pos[0]
            ax.scatter(i, df2.loc[i, 'p_mid'], s=200, marker='X', edgecolor='black', facecolor='white', label=lbl)
            ax.annotate(lbl, (i, df2.loc[i, 'p_mid']), textcoords="offset points", xytext=(5,5))

    ax.set_xlabel('Conjunto de prueba ordenado por p unificado (intervalo > ε)')
    ax.set_ylabel('Probabilidad')
    ax.set_xticks([])
    ax.legend(loc='upper left', fontsize=10, ncol=2)
    plt.tight_layout()
    sns.despine(bottom=True, left=True)
    plt.show()