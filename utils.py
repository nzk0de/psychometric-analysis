import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial, Gaussian
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import median_abs_deviation
import os
from typing import Tuple, Optional
from tabulate import tabulate
from scipy.stats import f_oneway
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem
import pandas as pd
from itertools import combinations
from scipy.stats import t
import numpy as np


def separate_directory_and_file(path: str) -> Tuple[str, Optional[str]]:
    if "." in os.path.basename(path):
        directory_path = os.path.dirname(path)
        file_name = os.path.basename(path)
    else:
        directory_path = path
        file_name = None
    return directory_path, file_name


def create_dirs(path: str) -> int:
    directory, _ = separate_directory_and_file(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        return 1
    else:
        return 0


######################################### 0 ###############################################
def print_mean(df, identifier):
    """
    Print:
    1. Mean RT and Accuracy for all rows
    2. Min and Max of the Session means

    Parameters:
    df (pd.DataFrame): DataFrame with 'RT', 'ACCURACY', and 'Session_Name_' columns
    """
    print(identifier)

    # Overall means (all rows)
    mean_rt = df["RT"].mean() / 1000  # Convert to seconds
    mean_accuracy = df["ACCURACY"].mean()
    print(f"Overall Mean RT: {mean_rt:.3f} seconds")
    print(f"Overall Mean Accuracy: {mean_accuracy:.3f}%")

    # Get mean accuracy for each session (but don't print them)
    session_means = df.groupby("Session_Name_")["ACCURACY"].mean()

    # Only print min and max of session means
    print(f"\nMin Session Mean Accuracy: {session_means.min():.3f}%")
    print(f"Max Session Mean Accuracy: {session_means.max():.3f}%")


######################################### 1.1 ###############################################


def preprocess_df(df):
    df = df[df["condition"] != "distractor"].reset_index(drop=True)
    df.rename(
        columns={
            "Session_Name_": "participant_id",
            "condition": "item_type",
            "ACCURACY": "accuracy",
            "RT": "reaction_time",
            "categoryofoperands": "group",
            "Trial_Index_": "trial_id",
        },
        inplace=True,
    )
    return df


def split_decimal_small(df):
    decimal_df = df[df["group"] == "decimal"].reset_index(drop=True)
    small_df = df[df["group"] == "small"].reset_index(drop=True)
    return decimal_df, small_df


######################################### 1.2 ###############################################
def analyze_reaction_times(
    df,
    participant_column="participant_id",
    reaction_time_column="reaction_time",
    method="median",
):
    """
    Analyze reaction times using either median/MAD or mean/STD method.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the reaction time data
    participant_column : str
        Name of the column containing participant IDs
    reaction_time_column : str
        Name of the column containing reaction times
    method : str
        Either 'median' or 'mean' to determine the analysis method

    Returns:
    --------
    tuple : (dict, tuple, pandas.DataFrame, str)
        Dictionary of results, tuple of (lower_bound, upper_bound), filtered dataframe, and method used
    """
    if method not in ["median", "mean"]:
        raise ValueError("Method must be either 'median' or 'mean'")

    # Calculate global statistics based on method
    if method == "median":
        central_tendency = df[reaction_time_column].median()
        dispersion = median_abs_deviation(df[reaction_time_column], scale="normal")
        central_name = "Median"
        dispersion_name = "MAD"
    else:  # mean
        central_tendency = df[reaction_time_column].mean()
        dispersion = df[reaction_time_column].std()
        central_name = "Mean"
        dispersion_name = "STD"

    # Define global valid range
    lower_bound = central_tendency - 3 * dispersion
    upper_bound = central_tendency + 3 * dispersion

    print(f"\nGlobal Statistics (using {method} method):")
    print(f"{central_name}: {central_tendency:.2f}")
    print(f"{dispersion_name}: {dispersion:.2f}")
    print(f"Valid range: ({lower_bound:.2f}, {upper_bound:.2f})")

    # Create global mask for filtered dataframe
    global_mask = (df[reaction_time_column] >= lower_bound) & (
        df[reaction_time_column] <= upper_bound
    )
    filtered_df = df[global_mask].copy()

    # Initialize results dictionary
    results = {}

    # Process each participant
    for participant in df[participant_column].unique():
        # Get participant data
        participant_data = df[df[participant_column] == participant][
            reaction_time_column
        ]

        # Create mask for valid trials using global bounds
        valid_mask = (participant_data >= lower_bound) & (
            participant_data <= upper_bound
        )
        valid_data = participant_data[valid_mask]

        # Calculate participant statistics based on method
        if method == "median":
            rt_central = valid_data.median() if valid_mask.sum() > 0 else np.nan
            rt_dispersion = (
                median_abs_deviation(valid_data, scale="normal")
                if valid_mask.sum() > 0
                else np.nan
            )
        else:  # mean
            rt_central = valid_data.mean() if valid_mask.sum() > 0 else np.nan
            rt_dispersion = valid_data.std() if valid_mask.sum() > 0 else np.nan

        # Store results with consistent key names
        results[participant] = {
            "total_trials": len(participant_data),
            "valid_trials": valid_mask.sum(),
            "valid_percentage": (valid_mask.sum() / len(participant_data)) * 100,
            "valid_mask": valid_mask,
            "central_rt": rt_central,  # Using consistent keys regardless of method
            "dispersion_rt": rt_dispersion,
        }

    return results, (lower_bound, upper_bound), filtered_df


def analyze_reaction_times(
    df,
    participant_column="participant_id",
    reaction_time_column="reaction_time",
    method="median",
):
    """
    Analyze reaction times using either median/MAD or mean/STD method.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the reaction time data
    participant_column : str
        Name of the column containing participant IDs
    reaction_time_column : str
        Name of the column containing reaction times
    method : str
        Either 'median' or 'mean' to determine the analysis method

    Returns:
    --------
    tuple : (dict, tuple, pandas.DataFrame, str)
        Dictionary of results, tuple of (lower_bound, upper_bound), filtered dataframe, and method used
    """
    if method not in ["median", "mean"]:
        raise ValueError("Method must be either 'median' or 'mean'")

    if method == "median":
        # Use 5th and 95th percentiles for bounds
        lower_bound = np.percentile(df[reaction_time_column], 5)
        upper_bound = np.percentile(df[reaction_time_column], 95)
        central_name = "Median"
        dispersion_name = "Interquantile Range"
    else:  # mean
        # Use mean and standard deviation for bounds
        central_tendency = df[reaction_time_column].mean()
        dispersion = df[reaction_time_column].std()
        lower_bound = central_tendency - 3 * dispersion
        upper_bound = central_tendency + 3 * dispersion
        central_name = "Mean"
        dispersion_name = "STD"

    print(f"\nGlobal Statistics (using {method} method):")
    if method == "median":
        print(f"Valid range: ({lower_bound:.2f}, {upper_bound:.2f})")
    else:
        print(f"{central_name}: {central_tendency:.2f}")
        print(f"{dispersion_name}: {dispersion:.2f}")
        print(f"Valid range: ({lower_bound:.2f}, {upper_bound:.2f})")

    # Create global mask for filtered dataframe
    global_mask = (df[reaction_time_column] >= lower_bound) & (
        df[reaction_time_column] <= upper_bound
    )
    filtered_df = df[global_mask].copy()

    # Initialize results dictionary
    results = {}

    # Process each participant
    for participant in df[participant_column].unique():
        # Get participant data
        participant_data = df[df[participant_column] == participant][
            reaction_time_column
        ]

        # Create mask for valid trials using global bounds
        valid_mask = (participant_data >= lower_bound) & (
            participant_data <= upper_bound
        )
        valid_data = participant_data[valid_mask]

        # Calculate participant statistics based on method
        if method == "median":
            rt_central = valid_data.median() if valid_mask.sum() > 0 else np.nan
            rt_dispersion = (
                np.percentile(valid_data, 95) - np.percentile(valid_data, 5)
                if valid_mask.sum() > 0
                else np.nan
            )
        else:  # mean
            rt_central = valid_data.mean() if valid_mask.sum() > 0 else np.nan
            rt_dispersion = valid_data.std() if valid_mask.sum() > 0 else np.nan

        # Store results with consistent key names
        results[participant] = {
            "total_trials": len(participant_data),
            "valid_trials": valid_mask.sum(),
            "valid_percentage": (valid_mask.sum() / len(participant_data)) * 100,
            "valid_mask": valid_mask,
            "central_rt": rt_central,  # Using consistent keys regardless of method
            "dispersion_rt": rt_dispersion,
        }

    return results, (lower_bound, upper_bound), filtered_df


def print_statistics(results, method):
    # Determine column headers based on method
    if method == "median":
        rt_header = "Median RT"
        dispersion_header = "MAD RT"
    else:  # mean
        rt_header = "Mean RT"
        dispersion_header = "STD RT"

    print("\nParticipant Statistics:")
    print("=" * 90)
    print(
        f"{'Participant':^12} | {'Total Trials':^12} | {'Valid Trials':^12} | {'Valid %':^12} | {rt_header:^12} | {dispersion_header:^12}"
    )
    print("-" * 90)

    for participant, stats in results.items():
        print(
            f"{participant:^12} | {stats['total_trials']:^12} | {stats['valid_trials']:^12} | "
            f"{stats['valid_percentage']:^12.2f} | {stats['central_rt']:^12.2f} | {stats['dispersion_rt']:^12.2f}"
        )


def plot_reaction_time_histograms_valid(
    df, results, bounds, method, title="Decimal", directory="results"
):
    participant_column = "participant_id"
    reaction_time_column = "reaction_time"
    participants = df[participant_column].unique()
    n_participants = len(participants)

    # Set up the grid
    cols = min(4, n_participants)
    rows = (n_participants // cols) + (n_participants % cols > 0)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows * cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # Plot for each participant
    for i, participant in enumerate(participants):
        participant_data = df[df[participant_column] == participant]
        valid_mask = results[participant]["valid_mask"]

        # Create histogram data
        hist_range = (
            participant_data[reaction_time_column].min(),
            participant_data[reaction_time_column].max(),
        )
        bins = np.linspace(hist_range[0], hist_range[1], 21)  # 20 bins

        # Plot both histograms using plt.hist
        axes[i].hist(
            participant_data[valid_mask][reaction_time_column],
            bins=bins,
            color="blue",
            alpha=0.5,
            label="Valid",
        )
        axes[i].hist(
            participant_data[~valid_mask][reaction_time_column],
            bins=bins,
            color="red",
            alpha=0.5,
            label="Invalid",
        )

        # Add vertical lines for bounds
        axes[i].axvline(
            bounds[0], color="black", linestyle="--", alpha=0.5, label="Bounds"
        )
        axes[i].axvline(bounds[1], color="black", linestyle="--", alpha=0.5)

        # Add vertical line for participant's central tendency
        if not np.isnan(results[participant]["central_rt"]):
            axes[i].axvline(
                results[participant]["central_rt"],
                color="green",
                linestyle="-",
                alpha=0.7,
                label="Median" if method == "median" else "Mean",
            )

        axes[i].set_title(f"Participant: {participant}")
        axes[i].set_xlabel("Reaction Time (ms)")
        axes[i].set_ylabel("Frequency")
        axes[i].legend()

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add a title to the entire figure
    fig.suptitle(title, fontsize=16)

    # Save the plot to the 'plots' directory with a filename based on the title
    sanitized_title = title.replace(" ", "_").replace(":", "").replace("/", "_")
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to include suptitle
    save_path = f"{sanitized_title}_outlier_{method}.png"
    save_path = os.path.join(directory, save_path)
    create_dirs(save_path)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")


######################################### 1.3 ###############################################


def perform_gee(
    df,
    dependent_var,
    group_var,
    analysis_type="linear",
    subject_col="Session_Name_",
):
    """
    Perform GEE analysis

    Parameters:
    -----------
    df : pandas DataFrame
        Input data
    dependent_var : str
        Name of the dependent variable column (e.g., 'ACCURACY' or 'RT')
    group_var : str
        Name of the grouping variable column (e.g., 'condition')
    analysis_type : str
        Type of analysis ('linear' for continuous data like RT, 'logistic' for binary data like accuracy)
    subject_col : str
        Name of the column containing subject identifiers

    Returns:
    --------
    GEEResults
        The fitted GEE model results
    """

    # Ensure numeric type for dependent variable
    df[dependent_var] = pd.to_numeric(df[dependent_var], errors="coerce")

    # Setup family based on analysis type
    if analysis_type == "linear":
        family = Gaussian()
    elif analysis_type == "logistic":
        family = Binomial()
    else:
        raise ValueError("analysis_type must be either 'linear' or 'logistic'")

    # Fit GEE model
    gee_model = smf.gee(
        formula="accuracy ~ C(item_type)",
        groups=df["participant_id"],
        data=df,
        family=family,
        cov_struct=sm.cov_struct.Independence(),
    )
    return gee_model.fit()


def pairwise_comparisons_with_bonferroni(model, comparisons, alpha=0.05):
    """
    Perform pairwise comparisons for a GEE model with Bonferroni adjustment.

    Parameters:
        model: Fitted GEE model (e.g., from statsmodels.gee.GEE)
        comparisons: List of tuples specifying terms to compare (e.g., [(term1, term2)]).
        alpha: Significance level for Bonferroni adjustment (default: 0.05).

    Returns:
        results: List of dictionaries with comparisons, odds ratios, p-values, and adjusted p-values.
    """
    coef = model.params
    cov_matrix = model.cov_params()

    results = []
    raw_p_values = []

    for term1, term2 in comparisons:
        # Compute the contrast
        contrast = coef[term1] - (coef[term2] if term2 in coef else 0)
        std_err = np.sqrt(
            cov_matrix.loc[term1, term1]
            + (cov_matrix.loc[term2, term2] if term2 in coef else 0)
            - 2 * (cov_matrix.loc[term1, term2] if term2 in coef else 0)
        )
        z_stat = contrast / std_err
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))
        odds_ratio = np.exp(contrast)
        conf_int = (
            np.exp(contrast - 1.96 * std_err),
            np.exp(contrast + 1.96 * std_err),
        )

        # Collect raw p-values for adjustment
        raw_p_values.append(p_value)

        results.append(
            {
                "Comparison": f"{term1} vs {term2}",
                "Odds Ratio": odds_ratio,
                "p-value": p_value,
                "95% CI": conf_int,
            }
        )

    # Apply Bonferroni adjustment
    adjusted_p_values = multipletests(raw_p_values, method="bonferroni", alpha=alpha)[1]

    # Add adjusted p-values to results
    for i, result in enumerate(results):
        result["Adjusted p-value"] = adjusted_p_values[i]

    return results


def plot_line_chart(
    df,
    title="Decimal",
    metric="accuracy",
    method="",
    directory="results",
    metric_name="Accuracy",
):

    # Group data and calculate mean and SEM
    grouped = df.groupby("item_type")[metric].agg(["mean", "sem"]).reset_index()

    # Ensure the correct order of item types
    # Print table of means and standard deviations
    print("\nSummary Table:")
    table = grouped[["item_type", "mean", "sem"]].round(3)
    print(tabulate(table, headers=["Item Type", "Mean", "SEM"], tablefmt="grid"))

    # Create the plot
    plt.figure(figsize=(12, 10))
    plt.errorbar(
        grouped["item_type"],
        grouped["mean"],
        yerr=grouped["sem"],
        fmt="-o",  # Line with circular markers
        capsize=5,
        alpha=0.7,
        label=f"Mean {metric_name} ± SEM",
    )
    plt.xlabel("Item Type")
    plt.ylabel(f"{metric_name}")
    plt.title(f"{metric_name} and Standard Error per Item Type for {title}")
    plt.grid(alpha=0.3)
    plt.legend()

    # Add a title to the figure
    plt.suptitle(title, fontsize=16)

    # Save the plot
    sanitized_title = title.replace(" ", "_").replace(":", "").replace("/", "_")

    save_path = f"{sanitized_title}_comparison_{metric}"
    save_path = save_path + f"_{method}.png" if method else f"{save_path}.png"
    save_path = os.path.join(directory, save_path)
    plt.tight_layout()
    create_dirs(save_path)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")


def perform_nway_anova_with_lsd(df, metric, category_columns, perform_lsd=False):
    """
    Perform N-way ANOVA with main and interaction effects, and LSD post-hoc testing.

    Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        metric (str): The name of the column containing the continuous variable (e.g., reaction time).
        category_columns (list): A list of categorical column names to analyze.

    Returns:
        None: Prints the results of the ANOVA test and LSD pairwise comparisons.
    """
    if not isinstance(category_columns, list):
        raise ValueError("category_columns should be a list of column names.")

    # Ensure categorical columns are treated as categorical variables
    for col in category_columns:
        df[col] = df[col].astype("category")

    # Build the formula for the ANOVA
    formula = f"{metric} ~ " + " + ".join(category_columns)
    formula += " + " + ":".join(category_columns)  # Add interaction terms

    # Fit the model
    model = smf.ols(formula, data=df).fit()

    # Perform ANOVA
    anova_table = sm.stats.anova_lm(model, typ=2)  # Type II sums of squares
    print(f"N-Way ANOVA Results for {metric} with {', '.join(category_columns)}:")
    print(anova_table)

    # Check significant effects
    significant_effects = anova_table[anova_table["PR(>F)"] < 0.05]
    if significant_effects.empty:
        print("\nNo significant effects found. Skipping LSD post-hoc testing.")
        return
    if not perform_lsd:
        return
    # LSD post-hoc testing
    print("\nPerforming LSD post-hoc testing...")
    # Calculate Mean Square Error (MSE)
    residuals = df[metric] - model.fittedvalues
    mse = np.sum(residuals**2) / model.df_resid

    # Residual degrees of freedom
    df_residual = model.df_resid

    # Pairwise comparisons
    combined_category = df[category_columns].astype(str).agg("-".join, axis=1)
    group_means = df.groupby(combined_category)[metric].mean()
    group_sizes = df.groupby(combined_category).size()

    results = []
    for group1, group2 in combinations(group_means.index, 2):
        # Mean difference
        mean_diff = group_means[group1] - group_means[group2]

        # LSD value
        t_critical = t.ppf(1 - 0.05 / 2, df_residual)
        lsd = t_critical * np.sqrt(
            mse * (1 / group_sizes[group1] + 1 / group_sizes[group2])
        )

        # Significance check
        significant = abs(mean_diff) > lsd

        # Save result
        results.append(
            {
                "Group 1": group1,
                "Group 2": group2,
                "Mean Difference": mean_diff,
                "LSD": lsd,
                "Significant": significant,
            }
        )

    # Display results
    results_df = pd.DataFrame(results)
    print("\nLSD Pairwise Comparisons:")
    print(results_df)

import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp
import warnings

def perform_kruskal_wallis_with_posthoc(df, metric, category_columns, perform_posthoc=True):
    """
    Performs the Kruskal-Wallis H-test for independent samples and optional post-hoc pairwise comparisons.

    Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        metric (str): The name of the column containing the continuous variable (e.g., reaction time).
        category_columns (list): A list of categorical column names to analyze.
        perform_posthoc (bool): Whether to perform post-hoc pairwise comparisons if the test is significant.

    Returns:
        None: Prints the results of the Kruskal-Wallis test and post-hoc comparisons if requested.
    """
    if not isinstance(category_columns, list):
        raise ValueError("category_columns should be a list of column names.")

    # Suppress warnings for better readability
    warnings.filterwarnings("ignore")

    for category in category_columns:
        if category not in df.columns:
            print(f"Column '{category}' not found in the dataframe. Skipping.")
            continue

        # Ensure the category column is treated as a categorical variable
        df[category] = df[category].astype("category")

        print(f"\nPerforming Kruskal-Wallis test for '{metric}' across '{category}' groups:")

        # Extract the groups
        groups = [group[metric].dropna() for name, group in df.groupby(category)]

        # Check if there are at least two groups
        if len(groups) < 2:
            print(f"Not enough groups in '{category}' to perform the test. Skipping.")
            continue

        # Perform the Kruskal-Wallis H-test
        stat, p_value = stats.kruskal(*groups)

        print(f"Kruskal-Wallis H-statistic: {stat:.4f}, p-value: {p_value:.4f}")

        # Determine significance
        if p_value < 0.05:
            print("Result: Significant differences detected.")
            if perform_posthoc:
                print("\nPerforming post-hoc pairwise comparisons using Dunn's test with Bonferroni correction:")
                try:
                    # Perform Dunn's test
                    dunn = sp.posthoc_dunn(df, val_col=metric, group_col=category, p_adjust='bonferroni')
                    print(dunn)
                except Exception as e:
                    print(f"An error occurred during post-hoc testing: {e}")
        else:
            print("Result: No significant differences detected. Skipping post-hoc testing.")

    # Reset warnings
    warnings.filterwarnings("default")
######################################### 2b ###############################################


def multiple_choice_accuracy_calc(df):
    df = df[df["accuracy"] == 1]
    df.drop(columns=["accuracy"], inplace=True)
    df["accuracy"] = (df["correctmultiplechoice"] == df["multipleresponse"]).astype(int)
    return df


######################################### 3 ###############################################
#                                     EYE_TRACKER                                         #
###########################################################################################
import pandas as pd
import re


class InterestAreaClassifier:
    def __init__(self, filepath):
        self.areas = {}
        self._parse_file(filepath)

    def _parse_file(self, filepath):
        """Parse the text file to extract coordinates and their categories."""
        mapping = {
            "EQUATION_INTERESTAREA": "equation",
            "SOLVABLE_INTERESTAREA": "linetai",
            "INSOLVABLE_INTERESTAREA": "denlinetai",
            "RESPONSE_A_INTERESTAREA": "a",
            "RESPONSE_B_INTERESTAREA": "b",
            "RESPONSE_C_INTERESTAREA": "c",
        }
        with open(filepath, "r") as file:
            for line in file:
                parts = line.split()
                if len(parts) >= 9:
                    x1, y1, x2, y2 = map(int, parts[3:7])
                    category = mapping.get(parts[7], None)
                    if category:
                        self.areas[category] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    def classify(self, x, y):
        """Classify a datapoint with coordinates (x, y) into a category."""
        for category, coords in self.areas.items():
            if coords["x1"] <= x <= coords["x2"] and coords["y1"] <= y <= coords["y2"]:
                return category
        return "Unknown"


def fix_condition(value):
    import re

    # Replace Greek characters with their Latin equivalents
    value = re.sub(r"Ι", "I", value)  # Replace Greek 'Ι' with Latin 'I'
    value = re.sub(r"Α", "A", value)  # Replace Greek 'Α' with Latin 'A'
    value = re.sub(r"Β", "B", value)  # Replace Greek 'Β' with Latin 'B'
    # Remove any other non-standard characters (optional)
    value = re.sub(r"[^a-zA-Z0-9]", "", value)
    return value
