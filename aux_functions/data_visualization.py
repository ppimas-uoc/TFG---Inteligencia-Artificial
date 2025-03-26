import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import math


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
        sns.histplot(data=df, x=col, kde=True, ax=axes[i], palette='viridis')
        sns.despine()
        axes[i].set_title(col, fontsize=16)
        axes[i].set_xlabel('')

    for j in range(len(cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
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
    sns.barplot(x=null_counts.index, y=null_counts.values, palette='viridis')
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
    plt.show()


def viz_dispersion(df):
    df_no_id = df.drop('id', axis=1)
    unique_counts = df_no_id.nunique()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    sns.boxplot(data=df_no_id, ax=axes[0], palette='viridis')
    axes[0].set_title("Dispersion", fontsize=14)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90)

    sns.despine(ax=axes[0])

    sns.lineplot(x=unique_counts.index, y=unique_counts.values, marker='o', ax=axes[1], palette='viridis')
    axes[1].set_title('Unique Values', fontsize=14)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90)

    axes[1].set_xlabel('')
    sns.despine(ax=axes[1])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    plt.show()


def viz_single_variable(df, column, target='QoL'):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))
    sns.histplot(
        data=df,
        x=column,
        hue=target,
        palette='viridis',
        multiple='fill',
        legend=False,
        ax=axes[0]
    )
    if axes[0].legend_:
        axes[0].legend_.remove()

    val_min, val_max = df[target].min(), df[target].max()
    norm = colors.Normalize(vmin=val_min, vmax=val_max)
    sm = cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar_left = plt.colorbar(sm, ax=axes[0])
    cbar_left.set_label(target)
    cbar_left.set_ticks([val_min, val_max])
    cbar_left.set_ticklabels([f"{val_min:.2f}", f"{val_max:.2f}"])
    axes[0].set_title(f"Distribution of {column} by {target}", fontsize=12)
    axes[0].set_xlabel(column, fontsize=10)
    axes[0].set_ylabel("Proportion", fontsize=10)

    sns.boxplot(data=df, x=target, y=column, ax=axes[1])
    axes[1].set_title(f"{column} vs {target}", fontsize=12)
    axes[1].set_xlabel('')
    axes[1].set_xticklabels([])
    axes[1].set_ylabel(column, fontsize=10)

    # Crear otra barra de color asociada a QoL en el eje del boxplot
    # (horizontal, debajo del gráfico)
    norm2 = colors.Normalize(vmin=val_min, vmax=val_max)
    sm2 = cm.ScalarMappable(cmap='viridis', norm=norm2)
    sm2.set_array([])
    cbar_right = plt.colorbar(sm2, ax=axes[1], orientation='horizontal')
    cbar_right.set_label(target)
    cbar_right.set_ticks([val_min, val_max])
    cbar_right.set_ticklabels([f"{val_min:.2f}", f"{val_max:.2f}"])

    plt.tight_layout()
    sns.despine()
    plt.subplots_adjust(wspace=0.2)
    plt.show()
    print('')
    print('')
    print(f"Statistics for {column} by {target} values:")
    print('')
    stats_table = df.groupby(target)[column].describe()
    display(stats_table)