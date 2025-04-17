import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import math
from itertools import combinations

from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import confusion_matrix, roc_curve

from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, \
    f1_score, make_scorer
from sklearn.model_selection import TunedThresholdClassifierCV


def viz_columns_distribution(df: pd.DataFrame):
    """
    Visualizes the distribution of columns present in a DataFrame.
    :param df: dataframe containing columns of interest
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
    Visualizes missing values of columns present in a DataFrame.
    :param df: dataframe containing columns of interest
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


def viz_split_distributions(y_list: [pd.Series], split_names: [str]):
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

    ax.set_title("Proportion of Each QoL Class in y datasets", fontsize=14)
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
    classes = [key for key in report_train.keys() if key not in ['accuracy', 'macro avg', 'weighted avg']]

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

    return y_pred_train, y_pred_val

def viz_confusion_matrix_thres(labels, y_val, y_pred_val, y_pred_thres):
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
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                xticklabels=labels, yticklabels=labels)
    plt.title("Matriz de confusión - Test", fontsize=14)
    plt.xlabel("QoL Predicha", fontsize=12)
    plt.ylabel("QoL Verdadera", fontsize=12)
    plt.tight_layout()
    plt.show()

def viz_threshold_behavior(model, X_val, y_val, num_thresholds=20):
    scoring = make_scorer(f1_score)
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

    def tipo_variable(col):
        if col in functioning:
            return "functioning"
        elif col in symptoms:
            return "symptom"
        else:
            return "other"

    tipos = [tipo_variable(col) for col in sorted_features]
    colors = {'functioning': '#66c2a5', 'symptom': '#fc8d62', 'other': '#8da0cb'}
    bar_colors = [colors[t] for t in tipos]

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

    titulo = "PDP + ICE para todas las variables" if kind == "both" else "PDP para todas las variables"
    fig.suptitle(titulo, fontsize=16)
    plt.tight_layout()
    fig.subplots_adjust(top=0.93)
    plt.show()


def viz_pdp_pairs(model, X, features, kind='average', cols=2):
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
