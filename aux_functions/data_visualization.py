import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import math
from itertools import combinations

from numpy.core.defchararray import upper
from pandas import DataFrame, Series


# Graficar la distribución de todas las variables del dataframe
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


def viz_split_distributions(y_list: [Series], split_names: [str]):
    data = []
    split_names = [f"{name} ({len(y)})" for name, y in zip(split_names, y_list)]
    for i, y in enumerate(y_list):
        proportions = y.value_counts(normalize=True).sort_index()
        for clase, prop in proportions.items():
            data.append({
                'Split': split_names[i],
                'Class': clase,
                'Proportion': prop
            })

    df_props = pd.DataFrame(data)

    df_pivot = df_props.pivot(index='Split', columns='Class', values='Proportion').fillna(0)

    df_pivot = df_pivot.reindex(split_names)

    n_clases = df_pivot.shape[1]
    palette = sns.color_palette("viridis", n_clases)

    ax = df_pivot.plot(
        kind='bar',
        stacked=True,
        color=palette,
        figsize=(1.5*len(split_names)+6, 5)
    )

    ax.set_title("Proportion of Each QoL Class in y datasets", fontsize=14)
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_xlabel("")
    ax.legend(title="QoL", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=0)

    plt.tight_layout()
    sns.despine()
    plt.show()


