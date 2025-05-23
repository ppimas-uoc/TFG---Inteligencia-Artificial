"""
data_visualization.py

Visualization utilities for exploratory data analysis, feature distribution,
model interpretation, and performance reporting.

Includes:
- Distribution and dispersion plots
- Histograms and ECDFs grouped by target
- Partial Dependence Plots (PDP) and ICE plots
- Feature importance (tree-based and permutation)
- Classification reports and confusion matrices
- Threshold tuning behavior visualization

Author: Pablo Pimàs Verge
Created: 2025-04
License:CC 3.0
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import math
from itertools import combinations
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, \
    f1_score, make_scorer, confusion_matrix
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss


def viz_columns_distribution(df: pd.DataFrame):
    """
    Plots the distribution (histograms with KDE) of all columns in the given DataFrame.

    :param df: DataFrame containing numerical or categorical variables to visualize.
    :return: None. Displays a grid of plots.
    """
    cols = df.columns

    n_cols = 5
    n_rows = math.ceil(len(cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.histplot(data=df, x=col, ax=axes[i], kde=True)
        sns.despine()
        axes[i].set_title(col, fontsize=16)
        axes[i].set_xlabel('')

    for j in range(len(cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    sns.despine()
    plt.subplots_adjust(hspace=0.5)
    plt.show()


# Visualizar valores nulos del dataframe por columna
def viz_missing_values(df):
    """
    Displays the number of missing values per column using a bar plot.

    :param df: DataFrame to analyze for missing values.
    :return: None. Shows a horizontal bar plot.
    """
    null_counts = df.isnull().sum()
    plt.figure(figsize=(12, 3))
    sns.barplot(x=null_counts.index, y=null_counts.values)
    sns.despine()
    plt.xticks(rotation=90)
    plt.ylabel('Number of Null Values')
    plt.title('Null Values per Variable')
    plt.show()


def viz_attr_histogram(df, variable, target='QoL'):
    """
    Plots the distribution of a single feature colored by the target class using a filled histogram.

    :param df: DataFrame containing the data.
    :param variable: Name of the feature to plot.
    :param target: Target column used for hue and color bar.
    :return: None. Displays the histogram and color scale.
    """
    plt.figure(figsize=(8, 4))
    ax = sns.histplot(
        data=df,
        x='Physical functioning',
        hue=target,
        palette='viridis',
        multiple='fill',
        legend=False
    )

    if ax.legend_:
        ax.legend_.remove()

    qol_min, qol_max = df[target].min(), df[target].max()
    norm = colors.Normalize(vmin=qol_min, vmax=qol_max)
    sm = cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(target)
    cbar.set_ticks([qol_min, qol_max])
    cbar.set_ticklabels([f"{qol_min:.2f}", f"{qol_max:.2f}"])
    plt.title("Histogram of Physical functioning by QoL")
    sns.despine()
    plt.show()


def viz_dispersion(df):
    """
    Plots boxplots of all columns and a line chart showing the number of unique values per column.

    :param df: DataFrame with features to visualize.
    :return: None. Displays two subplots.
    """
    unique_counts = df.nunique()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    sns.boxplot(data=df, ax=axes[0])
    axes[0].set_title("Dispersion", fontsize=14)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90)

    sns.despine(ax=axes[0])

    sns.lineplot(x=unique_counts.index, y=unique_counts.values, marker='o', ax=axes[1])
    axes[1].set_title('Unique Values', fontsize=14)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90)

    axes[1].set_xlabel('')
    sns.despine(ax=axes[1])

    plt.tight_layout()
    sns.despine()
    plt.subplots_adjust(wspace=0.1)
    plt.show()


def viz_single_variable(df, column, target='QoL'):
    """
    Generates three views for a single feature: histogram, boxplot and ECDF; grouped by target class.

    :param df: DataFrame containing the data.
    :param column: Feature column to analyze.
    :param target: Name of the target column.
    :return: None. Displays the plots and a summary stats table with outliers and skewness.
    """
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
    sns.histplot(
        data=df,
        x=column,
        hue=target,
        palette='viridis',
        multiple='fill',
        legend=False,
        bins=len(df[column].unique()),
        shrink=1,
        ax=axes[0]
    )
    if axes[0].legend_:
        axes[0].legend_.remove()

    val_min, val_max = df[target].min(), df[target].max()
    norm = colors.Normalize(vmin=val_min, vmax=val_max)
    sm = cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar_left = plt.colorbar(sm, ax=axes[0])
    cbar_left.set_label(target, labelpad=-35)
    cbar_left.set_ticks([val_min, val_max])
    cbar_left.set_ticklabels([f"{val_min:.2f}", f"{val_max:.2f}"])
    axes[0].set_title(f"Proportion of {target} by {column} values", fontsize=12)
    axes[0].set_xlabel(column, fontsize=10)
    axes[0].set_ylabel("Proportion", fontsize=10)

    norm2 = colors.Normalize(vmin=val_min, vmax=val_max)
    sm2 = cm.ScalarMappable(cmap='viridis', norm=norm2)
    sm2.set_array([])
    cbar_right = plt.colorbar(sm2, ax=axes[1], orientation='horizontal', pad=0.01)
    cbar_right.set_label(target, labelpad=-10)
    cbar_right.set_ticks([val_min, val_max])
    cbar_right.set_ticklabels([f"{val_min:.2f}", f"{val_max:.2f}"])
    sns.boxplot(data=df, x=target, y=column, ax=axes[1])
    axes[1].set_title(f"Distribution of {column} by {target} values", fontsize=12)
    axes[1].set_xlabel('')
    axes[1].set_xticklabels([])
    axes[1].set_ylabel(column, fontsize=10)

    sns.ecdfplot(data=df, x=column, ax=axes[2], palette='viridis', legend=False)
    axes[2].set_title(f"Cumulative distribution of {column}", fontsize=12)
    axes[2].set_xlabel(column, fontsize=10)
    axes[2].set_ylabel("Proportion", fontsize=10)


    plt.tight_layout()
    sns.despine()
    plt.subplots_adjust(wspace=0.2)
    plt.show()
    print('')
    print('')
    print(f"Statistics for {column} by {target} values:")
    print('')

    def count_outliers(series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        return outliers.count(), np.sort(np.round(outliers.dropna().unique(), 2))

    grouped = df.groupby(target)[column]
    outliers_info = grouped.apply(count_outliers)
    stats_table = df.groupby(target)[column].describe()
    stats_table['Outliers Count'] = outliers_info.apply(lambda x: x[0])
    stats_table['Outliers'] = outliers_info.apply(lambda x: x[1])
    skew_values = grouped.apply(lambda x: x.skew()).rename('skew')
    stats_table = stats_table.join(skew_values)
    display(stats_table)

def viz_single_vs_all(df, hue_var=None, n_cols=4, fig_width=5, fig_height=4):
    """
    Plots all pairwise lineplots between columns in the DataFrame.

    :param df: DataFrame with features to analyze.
    :param hue_var: Optional categorical variable used for coloring the lines.
    :param n_cols: Number of columns in the subplot grid.
    :param fig_width: Width of each subplot.
    :param fig_height: Height of each subplot.
    :return: None. Displays a grid of pairwise line plots.
    """
    pairs = list(combinations(df.columns.tolist(), 2))
    n_pairs = len(pairs)
    n_rows = math.ceil(n_pairs / n_cols)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(fig_width * n_cols, fig_height * n_rows))

    if n_pairs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (x_var, y_var) in enumerate(pairs):
        ax = axes[idx]
        sub_df = df[[x_var, y_var]].dropna().sort_values(x_var)
        if hue_var:
            sns.lineplot(data=sub_df, x=x_var, y=y_var, hue=sub_df[hue_var],
                         ax=ax)
        else:
            sns.lineplot(data=sub_df, x=x_var, y=y_var, ax=ax)

        ax.set_title(f"{y_var} vs {x_var}", fontsize=18)
        ax.set_xlabel(x_var, fontsize=16)
        ax.set_ylabel(y_var, fontsize=16)
        ax.tick_params(axis='both', labelsize=12)

    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    sns.despine()
    plt.show()


def viz_distributions_by_target(df, target, ncols=3):
    """
    Plots filled histograms of all features, grouped by the specified target variable.

    :param df: DataFrame containing the data.
    :param target: Column name of the target variable.
    :param ncols: Number of columns in the plot grid.
    :return: None. Displays the histogram grid with proportion by class.
    """
    variables = [col for col in df.columns if col != target]
    n = len(variables)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*4))
    axes = axes.flatten()
    for i, var in enumerate(variables):
        sns.histplot(
            data=df,
            x=var,
            hue=target,
            palette='viridis',
            multiple='fill',
            legend= (i==0),
            bins=len(df[var].unique()),
            shrink=1,
            ax=axes[i]
        )
        axes[i].set_title(f"Proportion of {target} by {var} values", fontsize=14)
        axes[i].set_xlabel(var, fontsize=12)
        axes[i].set_ylabel("Proportion", fontsize=12)

    for j in range(len(variables), len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='upper right', frameon=True)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor('black')
    fig.legend(handles, labels, loc='upper right')

    plt.tight_layout()
    sns.despine()
    plt.show()


def viz_split_distributions(y_list: [pd.Series], split_names: [str], var_title: str):
    """
    Displays a stacked bar plot showing the proportion of classes in each data split.

    :param y_list: List of Series containing target values from each split.
    :param split_names: List of names for each split (e.g. ['train', 'test', 'val']).
    :param: var_title: The variable to the title of the plot.
    :return: None. Displays a stacked bar plot with class proportions and labels.
    """
    data = []
    for name, y in zip(split_names, y_list):
        props = y.value_counts(normalize=True).sort_index()
        for clase, val in props.items():
            data.append({
                'Split': f"{name} ({len(y)})",
                'Class': clase,
                'Proportion': val
            })
    df_props = pd.DataFrame(data)
    df_pivot = df_props.pivot(index='Split', columns='Class', values='Proportion').fillna(0)

    palette = sns.color_palette("viridis", df_pivot.shape[1])
    ax = df_pivot.plot(kind='bar', stacked=True, color=palette, figsize=(1.5*len(y_list)+6, 5))

    ax.set_title(f"Proportion of Each {var_title} Class in y datasets", fontsize=14)
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_xlabel("")
    ax.legend(title="QoL", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10)
    plt.xticks(rotation=0)

    xticks = ax.get_xticks()
    for i, split in enumerate(df_pivot.index):
        bottom = 0
        for clase in df_pivot.columns:
            val = df_pivot.loc[split, clase]
            if val > 0:
                ax.text(
                    xticks[i],
                    bottom + val/2,
                    f"{val*100:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="black"
                )
            bottom += val

    plt.tight_layout()
    sns.despine()
    plt.show()



def viz_classification_reports(model, X_train, y_train, X_val, y_val, val='validation'):
    """
    Prints and plots classification report metrics (precision, recall, F1) by class for train and validation.

    :param model: Trained classifier model with a `.predict()` method.
    :param X_train: Training features.
    :param y_train: Training target.
    :param X_val: Validation features.
    :param y_val: Validation target.
    :param val: Label for the validation set.
    :return: Tuple of predictions (y_pred_train, y_pred_val).
    """
    y_pred_train = model.predict(X_train)
    report_train = classification_report(y_train, y_pred_train, output_dict=True)
    print("Reporte sobre el conjunto de ENTRENAMIENTO:")
    print(classification_report(y_train, y_pred_train))

    y_pred_val = model.predict(X_val)
    report_val = classification_report(y_val, y_pred_val, output_dict=True)
    print(f"Reporte sobre el conjunto de {val}: ", val.upper())
    print(classification_report(y_val, y_pred_val))

    reports = {
        "entrenamiento": report_train,
        f"{val}": report_val,
    }

    ds_names = list(reports.keys())
    metrics = ['precision', 'recall', 'f1-score']
    classes = sorted(
        [k for k in report_train.keys()
         if k not in ['accuracy','macro avg','weighted avg']],
        key=float
    )

    plot_data = {metric: {cls: [] for cls in classes} for metric in metrics}
    for ds in ds_names:
        for cls in classes:
            for metric in metrics:
                plot_data[metric][cls].append(reports[ds][cls][metric])

    plt.figure(figsize=(15, 5))
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        for cls in classes:
            plt.plot(ds_names, plot_data[metric][cls], marker='o', label=f'Clase {cls}')
        plt.xlabel("Conjunto", fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.title(f"{metric.capitalize()} por clase en cada conjunto", fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True)
    plt.tight_layout()
    sns.despine()
    plt.show()

    return y_pred_train, y_pred_val, report_val

def viz_confusion_matrix_thres(labels, y_val, y_pred_val, y_pred_thres):
    """
    Displays side-by-side confusion matrices for original and thresholded predictions.

    :param labels: Class labels to annotate the matrix.
    :param y_val: Ground truth target values.
    :param y_pred_val: Predictions from model with default threshold.
    :param y_pred_thres: Predictions with custom threshold.
    :return: None. Displays two heatmaps.
    """
    cm_val = confusion_matrix(y_val, y_pred_val)
    cm_thres = confusion_matrix(y_val, y_pred_thres)
    matrices = [("Validación", cm_val), ("Validación Ajustada", cm_thres)]

    fig, axes = plt.subplots(1, ncols = 2, figsize=(12, 5))

    for ax, (title, cm) in zip(axes, matrices):
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                    xticklabels=labels, yticklabels=labels,
                    ax=ax)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("QoL Predicha", fontsize=12)
        ax.set_ylabel("QoL Verdadera", fontsize=12)

    plt.tight_layout()
    plt.show()

def viz_confusion_matrix_test(labels, y_true, y_pred):
    """
    Displays a confusion matrix for test set predictions.

    :param labels: Class labels to annotate axes.
    :param y_true: Ground truth values.
    :param y_pred: Predicted values.
    :return: None. Displays a single heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                xticklabels=labels, yticklabels=labels)
    plt.title("Matriz de confusión - Test", fontsize=14)
    plt.xlabel("QoL Predicha", fontsize=12)
    plt.ylabel("QoL Verdadera", fontsize=12)
    plt.tight_layout()
    plt.show()

def viz_threshold_behavior(model, X_val, y_val, num_thresholds=20, metric=f1_score):
    """
    Plots model performance metrics across different classification thresholds and shows the optimal one.

    :param model: Trained classifier with predict_proba.
    :param X_val: Validation features.
    :param y_val: Validation labels.
    :param num_thresholds: Number of threshold points to evaluate.
    :return: Tuple (best_threshold, df_metrics) with optimal F1 threshold and full score DataFrame.
    """
    scoring = make_scorer(metric)
    tuned_clf = TunedThresholdClassifierCV(model, cv="prefit", refit=False, scoring=scoring)
    tuned_clf.fit(X_val, y_val)
    best_threshold = tuned_clf.best_threshold_
    best_score = tuned_clf.best_score_

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_pred_proba)

    thresholds = np.linspace(0, 1, num_thresholds)
    metrics = []
    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        metrics.append((t, acc, prec, rec, f1))

    df_metrics = pd.DataFrame(metrics, columns=['Threshold', 'Accuracy', 'Precision', 'Recall', 'F1-score'])

    plt.figure(figsize=(12, 6))
    palette = sns.color_palette('viridis', n_colors=4)
    for col, color in zip(['Accuracy', 'Precision', 'Recall', 'F1-score'], palette):
        plt.plot(df_metrics['Threshold'], df_metrics[col], color=color, lw=2, label=col)
        plt.scatter(df_metrics['Threshold'], df_metrics[col], color=color, s=50, lw=1.5, edgecolor='white', zorder=12)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.plot(best_threshold, best_score, marker=3, markersize=10, color="#D55E00", label=f"F1 score óptimo con umbral = {best_threshold:.3f}")
    plt.plot([], [], ' ', label=f'ROC AUC: {roc_auc:.3f}')
    plt.legend(loc='lower left', ncol=3, fontsize=12)
    plt.title("Comportamiento del modelo por umbral de decisión", fontsize=14)
    plt.xlabel("Umbral de decisión", fontsize=12)
    plt.ylabel("Puntaje", fontsize=12)
    plt.tight_layout(pad=1)
    sns.despine()
    plt.show()

    return best_threshold, df_metrics

def viz_feature_importance(sorted_feat_imp: zip):
    """
    Plots horizontal bar chart of feature importances with color-coded variable types.

    :param sorted_feat_imp: Zip of (feature_name, importance) sorted from low to high.
    :return: None. Displays the bar chart.
    """
    sorted_features, sorted_importances = zip(*sorted_feat_imp)

    functioning = [
        'Physical functioning', 'Role functioning', 'Emotional functioning',
        'Cognitive functioning', 'Social functioning', 'Sexual functioning'
    ]

    symptoms = [
        'Fatigue', 'Nausea and vomiting', 'Pain', 'Dyspnea', 'Insomnia',
        'Appetite loss', 'Constipation', 'Diarrhea',
        'Arm symptoms', 'Breast symptoms', 'Systemic therapy side effects',
    ]

    def variable_type(col):
        if col in functioning:
            return "functioning"
        elif col in symptoms:
            return "symptom"
        else:
            return "other"

    types = [variable_type(col) for col in sorted_features]
    colors = {'functioning': '#66c2a5', 'symptom': '#fc8d62', 'other': '#8da0cb'}
    bar_colors = [colors[t] for t in types]

    plt.figure(figsize=(15, 5))
    plt.barh(range(len(sorted_features)), sorted_importances, color=bar_colors)
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel("Importance")
    plt.title("Feature Importance by variable type")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    labels = list(colors.keys())
    plt.legend(handles, labels, title="Type", loc="lower right")

    plt.tight_layout()
    sns.despine()
    plt.show()

def viz_pdp_single(model, X, kind='average', grid_resolution=101, n_cols=5):
    """
    Plots individual Partial Dependence Plots (and ICE if requested) for all features.

    :param model: Trained estimator.
    :param X: Dataset used for PDP computation.
    :param kind: 'average' for PDP, 'both' for PDP+ICE.
    :param grid_resolution: Number of grid points used for plotting.
    :param n_cols: Number of columns in the subplot grid.
    :return: None. Displays the grid of PDPs.
    """
    features = X.columns.tolist()
    n_features = len(features)
    n_rows = -(-n_features // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        disp = PartialDependenceDisplay.from_estimator(
            model,
            X,
            features=[feature],
            kind=kind,
            grid_resolution=grid_resolution,
            ax=axes[i]
        )
        ax = disp.axes_[0, 0]
        ax.set_ylim(0, 1)
        ax.set_title(feature, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    title = "PDP + ICE para todas las variables" if kind == "both" else "PDP para todas las variables"
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    fig.subplots_adjust(top=0.93)
    plt.show()


def viz_pdp_pairs(model, X, features, kind='average', cols=2):
    """
    Plots 2D PDP plots for selected feature pairs.

    :param model: Trained model.
    :param X: Dataset to compute PDPs on.
    :param features: List of 2-tuples representing feature pairs.
    :param kind: PDP plot kind, 'average' or 'both'.
    :param cols: Number of columns in the plot grid.
    :return: None. Displays PDP pair plots.
    """
    n = len(features)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()

    display = PartialDependenceDisplay.from_estimator(
        model,
        X,
        features=features,
        kind=kind,
        random_state=73,
        ax=axes[:n],
    )

    for ax in axes[:n]:
        for coll in ax.collections:
            if hasattr(coll, 'set_cmap'):
                coll.set_cmap('viridis')

    # Eliminar subplots vacíos si los hay
    for i in range(n, len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle("Gráficos PDP (2-way) por par de variables", fontsize=16)
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.show()


def viz_model_comparison(reports_dict):
    """
    Plots three side-by-side bar charts (precision, recall, and F1-score) per class for each model.

    Parameters:
        reports_dict: dict
            A dictionary where the keys are model names (str) and the values are classification reports
            generated with sklearn.metrics.classification_report(..., output_dict=True).
            Each report must include class-wise metrics with string class labels (e.g., '0.0', '1.0', '2.0').

    Returns:
        None. Displays the plots.
    """
    records = []
    for model_name, report in reports_dict.items():
        for cls in ['0.0', '1.0', '2.0']:  # ajusta según tus etiquetas
            metrics = report[cls]
            records.append({'Modelo': model_name, 'Clase': cls, 'Métrica': 'Precision', 'Valor': metrics['precision']})
            records.append({'Modelo': model_name, 'Clase': cls, 'Métrica': 'Recall', 'Valor': metrics['recall']})
            records.append({'Modelo': model_name, 'Clase': cls, 'Métrica': 'F1-score', 'Valor': metrics['f1-score']})

    df = pd.DataFrame.from_records(records)

    g = sns.catplot(
        data=df, kind='bar',
        x='Clase', y='Valor', hue='Modelo',
        col='Métrica', col_order=['Precision', 'Recall', 'F1-score'],
        height=4, aspect=1
    )

    g.set_titles('{col_name}')
    g.set_axis_labels('Clase', 'Valor')
    g.set(ylim=(0, 1))
    g.despine(left=True)
    plt.tight_layout()
    plt.show()

def viz_threshold_optimization(probs, y_val, y_pred_proba, threshold):

    df_plot = pd.DataFrame({
        'prob': probs,
        'verdadera': y_val,
        'predicha': y_pred_proba,
    })

    # Clasificación correcta o incorrecta
    df_plot['acierto'] = df_plot['verdadera'] == df_plot['predicha']
    df_plot['clase'] = df_plot['verdadera'].map({0: 'Aceptable', 1: 'Mejorable'})
    df_plot['acierto_str'] = df_plot['acierto'].map({True: 'Correcta', False: 'Incorrecta'})

    # Posición vertical con jitter para mejor visualización
    np.random.seed(42)
    df_plot['y'] = df_plot['verdadera'] + (np.random.rand(len(df_plot)) - 0.5) * 0.3

    # Colores y estilos
    palette = {'Correcta': '#1b9e77', 'Incorrecta': '#d95f02'}
    markers = {'Aceptable': 'o', 'Mejorable': 's'}

    plt.figure(figsize=(10, 3))
    # Redefinir colores por clase verdadera, # Azul = aceptable, Rojo = mejorable
    palette = {0: '#1f77b4', 1: '#d62728'}
    # Añadir columna de error (TP, FP, FN, TN)
    df_plot['tipo_error'] = df_plot.apply(lambda row:
                                          'TP' if row['verdadera'] == 1 and row['predicha'] == 1 else
                                          'TN' if row['verdadera'] == 0 and row['predicha'] == 0 else
                                          'FP' if row['verdadera'] == 0 and row['predicha'] == 1 else
                                          'FN', axis=1)
    sns.scatterplot(
        data=df_plot,
        x='prob',
        y='verdadera',
        hue='tipo_error',
        palette={'TP': '#2ca02c', 'TN': '#1f77b4', 'FP': '#ff7f0e', 'FN': '#d62728'},
        style='tipo_error',
        alpha=0.9,
        s=80
    )
    plt.axvline(x=threshold, color='black', linestyle='--', label=f'Umbral óptimo = {threshold:.3f}')
    plt.yticks([0, 1], ['Aceptable', 'Mejorable'])
    plt.ylabel('Clase verdadera')
    plt.xlabel('Probabilidad heurística o raw value (clase mejorable)')
    plt.title('Visualización de errores según el umbral de decisión')
    plt.legend(title='Clasificación')
    sns.despine()
    plt.tight_layout()
    plt.show()


def viz_calibration_curve(h_probs, y_true, n_bins=10, strategy='uniform'):
    """
       Plot calibration curves for one or more sets of predicted probabilities.

       Parameters
       ----------
       h_probs : list of tuple
           Each tuple is (probs, label), where `probs` is a 1d array of predicted probabilities
           for the positive class, and `label` is a string naming the method.
       y_true : array-like
           True binary labels (0/1).
       n_bins : int
           Number of probability bins.
       strategy : {'uniform', 'quantile'}
           Binning strategy passed to sklearn.calibration.calibration_curve.
       """
    plt.figure(figsize=(12, 6))
    for probs, label in h_probs:
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy=strategy)
        plt.plot(mean_pred, frac_pos, marker='o', markersize=6, linewidth=1.5, label=label)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Calibración perfecta')
    plt.xlabel('Probabilidad media predicha')
    plt.ylabel('Proporción de positivos')
    plt.title(f'Curva de calibración ({strategy}, bins={n_bins})')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    sns.despine(bottom=True, left=True)
    plt.show()