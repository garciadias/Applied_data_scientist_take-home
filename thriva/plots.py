import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
from matplotlib import patheffects
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# create color palette from list of colors
color_list = ["#fa476f", "#45d0eb", "#664277", "#f9d423"]
custom_color_cycle = cycler(color=color_list)


def percentages_calculation(df, col, target_col):
    df_defined = df[[col, target_col]].dropna()
    percentages = df_defined[[col, target_col]].groupby(col).value_counts().unstack()
    percentages = 100 * (percentages.T / percentages.sum(axis=1)).T[["Yes", "No"]]
    df_defined = df[[col, target_col]].dropna()
    percentages = df_defined[[col, target_col]].groupby(col).value_counts().unstack()
    percentages = 100 * (percentages.T / percentages.sum(axis=1)).T[["Yes", "No"]]
    return percentages


def global_yes_ratio(df, target_col):
    return df[target_col].value_counts()["Yes"] / len(df)


def stacked_bar_plot(df, col, target_col, ax, width=0.9):
    percentages = percentages_calculation(df, col, target_col)
    ax = percentages.plot(
        kind="bar", stacked=True, ax=ax, width=width, color=["#fa476f", "#45d0eb"]
    )
    ax = write_yes_no_percentages_to_bars(ax, percentages)
    return ax


def write_yes_no_percentages_to_bars(ax, percentages):
    # show percentages on top of bars
    for j, (_, (no, yes)) in enumerate(percentages.iterrows()):
        x_pos = j - 0.3
        y_no_pos = no
        y_yes_pos = no + yes

        ax.text(
            s=f"{int(round(no, 0)):d}",
            x=x_pos,
            y=y_no_pos,
            fontsize=18,
            color="w",
            fontweight="bold",
            path_effects=[patheffects.withStroke(linewidth=3, foreground="k")],
        )
        # shows text with colored edge
        ax.text(
            s=f"{int(round(yes, 0)):d}",
            x=x_pos,
            y=y_yes_pos,
            fontsize=18,
            fontweight="bold",
            color="w",
            path_effects=[patheffects.withStroke(linewidth=3, foreground="k")],
        )
    return ax


def hline_plot(df, target_col, ax):
    xmin, xmax = ax.get_xlim()
    yes_ratio = global_yes_ratio(df, target_col)
    # show global percentage of yes
    ax.hlines(
        yes_ratio * 100,
        xmin=xmin,
        xmax=xmax,
        linestyles="dashed",
        linewidth=5,
        label="Global Yes Ratio",
        color="#664277",
    )
    return ax


def apply_style_to_plot(
    df, col, ax, first_col=False, legend_title="Vitamin D\nSupplement"
):
    # cast xlabels to integer
    if df[col].dtype == "category":
        ax.tick_params(axis="x", rotation=90, labelsize=18)
    else:
        ax.set_xticklabels(
            [int(x) if isinstance(x, float) else str(x) for x in ax.get_xticks()],
            rotation=0,
            fontsize=18,
        )
    ax.set_xlabel("", fontsize=0)
    ax.set_title(col, fontsize=20)
    try:
        ax.get_legend().remove()
    except AttributeError:
        # no legend to remove
        print(type(ax))
    if first_col:
        ax.legend(
            loc="lower left",
            fontsize=18,
            title=legend_title,
            title_fontsize=18,
            framealpha=1,
        )
        ax.set_ylabel("Percentages", fontsize=20)
        ax.tick_params(axis="y", labelsize=18)
    else:
        ax.tick_params(axis="y", labelsize=0)
        ax.set(ylabel=None)
    return ax


def plot_categorical_percentages(df, col_list, target_col, bottom=0.13):
    width = 0.9
    fig, axis = plt.subplots(1, 3, figsize=(16, 9))
    for i, (col, ax) in enumerate(zip(col_list, axis.flatten())):
        ax = stacked_bar_plot(df, col, target_col, ax, width=width)
        ax = hline_plot(df, target_col, ax)
        ax = apply_style_to_plot(df, col, ax, first_col=(i == 0))
    plt.subplots_adjust(
        hspace=0.4, wspace=0.0, left=0.06, right=0.999, bottom=bottom, top=0.96
    )
    return fig


def kde_plot(df, col_list, target_col, legend_title="Vitamin D\nSupplement"):
    fig, axis = plt.subplots(1, 2, figsize=(16, 9))
    for i, (col, ax) in enumerate(zip(col_list, axis.flatten())):
        sns.kdeplot(
            data=df,
            x=col,
            hue=target_col,
            common_norm=False,
            fill=True,
            alpha=0.25,
            linewidth=2,
            ax=ax,
            palette=["#45d0eb", "#fa476f"],
        )
        ax.set_title(col, fontsize=20)
        ax.set_xlabel("", fontsize=0)
        ax.set_ylabel("Density", fontsize=20)
        ax.tick_params(labelsize=18)
        if df.dtypes[col] == "datetime64[ns]":
            ax.tick_params(axis="x", rotation=45, labelsize=18)
            counts = df[col].value_counts()
            small_counts = counts[counts < 5].index
            min_date = df[~df[col].isin(small_counts)][col].min()
            max_date = df[col].max()
            ax.set_xlim(min_date, None)
        if i == 0:
            leg = ax.get_legend()
            leg.set_title(legend_title)
            leg.get_title().set_fontsize(18)
            for text in leg.get_texts():
                text.set_fontsize(18)
        else:
            ax.get_legend().remove()
            ax.set(ylabel=None)
        plt.subplots_adjust(wspace=0.13, left=0.07, right=0.999, bottom=0.05, top=0.96)
    return fig


def bar_plot(df, col, target_col, ax, width=0.9):
    # plot bar chart
    percentages = percentages_calculation(df, col, target_col)["Yes"]
    ax = percentages.plot(
        kind="bar", stacked=False, ax=ax, width=width, color=["#fa476f"]
    )
    write_yes_percentages_to_bars(ax, percentages)
    return ax


def write_yes_percentages_to_bars(ax, percentages):
    # show percentages on top of bars
    for j, (_, yes) in enumerate(percentages.items()):
        x_pos = j - 0.3
        # shows text with colored edge
        ax.text(
            s=f"{int(round(yes, 0)):d}",
            x=x_pos,
            y=yes,
            fontsize=18,
            fontweight="bold",
            color="w",
            path_effects=[patheffects.withStroke(linewidth=3, foreground="k")],
        )
    return None


def plot_hist_percentages(
    df, col_list, target_col, bottom=0.13, legend_title="Vitamin D < 50 nmol/L"
):
    width = 0.9
    fig, axis = plt.subplots(1, 3, figsize=(16, 9))
    for i, (col, ax) in enumerate(zip(col_list, axis.flatten())):
        ax = bar_plot(df, col, target_col, ax, width=width)
        ax = hline_plot(df, target_col, ax)
        ax = apply_style_to_plot(
            df, col, ax, first_col=(i == 0), legend_title=legend_title
        )
    plt.subplots_adjust(
        hspace=0.4, wspace=0.0, left=0.06, right=0.999, bottom=bottom, top=0.96
    )
    return fig


def plot_roc_curve(models, X_test, y_test, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        ax.set_prop_cycle(custom_color_cycle)
    for model in models.models:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred_proba > 0.5)
        ax.plot(
            fpr,
            tpr,
            lw=2,
            label=f"{model.__class__.__name__}\n(acc = {accuracy:.3f}, auc = {roc_auc:.3f})",
        )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("False Positive Rate", fontsize=20)
    ax.set_ylabel("True Positive Rate", fontsize=20)
    ax.tick_params(labelsize=18)
    ax.legend(loc="lower right", fontsize=18)
    plt.tight_layout()
    return fig
