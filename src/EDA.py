import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency


def compute_cohens_d(x1, x2):
    """
    Compute Cohen's d to estimate the effect size between two distributions.

    Args:
        x1 (array-like): First sample of numerical values.
        x2 (array-like): Second sample of numerical values.

    Returns:
        float: Cohen's d value indicating the standardized mean difference.
    """
    n1, n2 = len(x1), len(x2)
    s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1)*s1 + (n2 - 1)*s2) / (n1 + n2 - 2))
    d = (np.mean(x1) - np.mean(x2)) / s_pooled
    return d


def cramers_v(confusion_matrix):
    """
    Calculate Cramér's V to measure the association strength between two categorical variables.

    Args:
        confusion_matrix (DataFrame or ndarray): Contingency table of observed frequencies.

    Returns:
        float: Cramér's V statistic ranging from 0 (no association) to 1 (strong association).
    """
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))


def plot_categorical_analysis(var, df, title=None):
    """
    Plot the distribution and success rate of a categorical variable, and perform statistical analysis.

    The function displays two plots side by side:
    - A count plot showing the distribution of the variable.
    - A bar plot showing the mean success rate by category.

    It also computes and prints the Chi-squared p-value and Cramér's V effect size.

    Args:
        var (str): Name of the categorical variable to analyze.
        df (DataFrame): Data containing the variable and the target 'success'.
        title (str, optional): Custom title for the success rate plot. Defaults to auto-generated.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Univariate analysis
    sns.countplot(x=var, data=df, ax=axes[0], palette='pastel', hue=var)
    axes[0].set_title(f"Distribution of {var}")
    axes[0].set_ylabel("Segments Number")
    axes[0].set_xlabel(var)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', linestyle='--', alpha=0.3)
    if axes[0].get_legend():
        axes[0].get_legend().remove()

    # Bivariate plots
    temp = df.groupby(var)['success'].mean().reset_index()
    sns.barplot(x=var, y='success', data=temp, palette='viridis', ax=axes[1], hue=var)
    axes[1].set_title(title if title else f"Success rate according to {var}")
    axes[1].set_ylabel("Success rate")
    axes[1].set_xlabel(var)
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', linestyle='--', alpha=0.3)
    if axes[1].get_legend():
        axes[1].get_legend().remove()

    plt.tight_layout()
    plt.show()

    # Statistical tests
    contingency_table = pd.crosstab(df[var], df['success'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi² p-value for {var}: {p:.4f}")

    cramer_v_value = cramers_v(contingency_table)
    print(f"Cramér's V for {var}: {cramer_v_value:.3f}")