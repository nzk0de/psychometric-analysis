import pandas as pd
from utils import (
    InterestAreaClassifier,
    preprocess_df,
    split_decimal_small,
    perform_nway_anova_with_lsd,
    fix_condition,
)


# %%
def get_df(exp_num, type="theo"):
    filepath = f"exp{exp_num[0]}_{type}.xlsx"
    df = pd.read_excel(filepath)
    df = preprocess_df(df)
    return df


def filter_and_calculate_percentages(df, column="category", exclude_values=None):
    """
    Filters out rows with specified values in the given column and calculates percentages of remaining values.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to filter and analyze.
    column : str
        The name of the column to filter on.
    exclude_values : list or None
        Values to exclude from the column. If None, no rows are excluded.

    Returns:
    --------
    pandas.Series
        A Series of percentages for each unique value in the column after filtering.
    """
    if exclude_values is None:
        exclude_values = []

    # Filter out rows with the specified values
    filtered_df = df[~df[column].isin(exclude_values)].reset_index(drop=True)

    # Calculate counts and percentages
    value_counts = filtered_df[column].value_counts()
    percentages = value_counts / len(filtered_df)

    return percentages


# %%
# Example usage

# Initialize the classifier
classifier = InterestAreaClassifier("data/eye_tracker/101/aoi/IA_1.ias")
fixations_df = pd.read_excel("all_fixations.xlsx")
fixations_df["category"] = fixations_df.apply(
    lambda row: classifier.classify(row["axp"], row["ayp"]), axis=1
)
results_df_theo = get_df("2", "theo")
df_theo = results_df_theo.merge(
    fixations_df, on=["participant_id", "trial_id"], how="inner"
)

results_df_thet = get_df("2", "thet")
df_thet = results_df_thet.merge(
    fixations_df, on=["participant_id", "trial_id"], how="inner"
)


# %%

# Apply the classifier to the dataframe


# %%
# Select only categorical columns
############ All Fixations ############
for idx, df in enumerate([df_theo, df_thet]):
    title = "Theoritikoi" if idx == 0 else "Thetikoi"
    print(
        f"\n\n############################################# {title} #############################################"
    )
    category_counts = filter_and_calculate_percentages(df, exclude_values=None)
    print(category_counts)
    category_counts = filter_and_calculate_percentages(df, exclude_values=["Unknown"])
    print(category_counts)
    category_counts = filter_and_calculate_percentages(
        df, exclude_values=["Unknown", "denlinetai", "linetai"]
    )
    print(category_counts)
    category_counts = filter_and_calculate_percentages(
        df, exclude_values=["Unknown", "denlinetai", "linetai", "equation"]
    )
    print(category_counts)


# %%
# Count of each participant_id

pd.set_option("display.max_rows", None)
for idx, df in enumerate([df_theo, df_thet]):
    decimal_df, small_df = split_decimal_small(df)
    for idy, df in enumerate([decimal_df, small_df]):
        # Apply the function to the condition column
        df["item_type"] = df["item_type"].apply(fix_condition)
        # Generate the title
        base_title = "Theoritikoi" if idx == 0 else "Thetikoi"
        subtype = "Decimal" if idy == 0 else "Small"
        title = f"{base_title} - {subtype}"
        print(
            f"\n\n############################################# {title} #############################################"
        )
        participant_counts = df["participant_id"].value_counts()

        # Count of each participant_id, condition pair
        participant_condition_counts = df.groupby(
            ["participant_id", "item_type"]
        ).size()

        # Convert the results to DataFrames for better display (optional)
        participant_counts_df = participant_counts.reset_index(name="count").rename(
            columns={"index": "participant_id"}
        )
        participant_condition_counts_df = participant_condition_counts.reset_index(
            name="count"
        )

        # Display the results
        print("Count of each participant_id:")
        print(participant_counts_df)

        print("\nCount of each participant_id, condition pair:")
        print(participant_condition_counts_df)

        # Calculate the mean of counts for each participant_id
        mean_participant_count = participant_counts_df["count"].mean()

        # Display the results
        print("Mean count of occurrences per participant_id:")
        print(mean_participant_count)

        # Aggregate counts by condition and compute the mean for each condition
        condition_mean_counts = participant_condition_counts_df.groupby("item_type")[
            "count"
        ].mean()

        # Display the results
        print("Mean count of occurrences per condition:")
        print(condition_mean_counts)

        print("Mean count of occurrences per condition category participant pair")
        # Filter for relevant categories
        filtered_df = df[df["category"].isin(["a", "b", "c"])]
        grouped_counts = (
            filtered_df.groupby(["participant_id", "item_type", "category"])
            .size()
            .reset_index(name="count")
        )

        # Step 2: Calculate the mean count for each item_type and category across participants
        mean_counts = (
            grouped_counts.groupby(["item_type", "category"])["count"]
            .mean()  # Compute the mean count for each item_type-category pair across participants
            .reset_index(name="mean_count")
        )

        # Output
        print(mean_counts)
        # # Calculate total counts per condition
        # total_counts = grouped_counts.groupby("item_type")["count"].transform("sum")

        # # Calculate percentages
        # grouped_counts["percentage"] = (grouped_counts["count"] / total_counts) * 100

        # # Display results
        # print(grouped_counts.mean())

        print(
            "######################## ANOVA Test for item_type-category pairs ########################"
        )
        group_columns = ["participant_id", "trial_id", "item_type", "category"]
        # Calculate percentages
        # iNCLUDE ONLY ABC
        grouped_counts = (
            filtered_df.groupby(group_columns).size().reset_index(name="counts")
        )
        # print(grouped_counts)
        perform_nway_anova_with_lsd(
            grouped_counts, "counts", ["item_type", "category"], perform_lsd=True
        )

        # perform_nway_anova_with_lsd(grouped_counts, "counts", ["item_type"])
        print(
            "######################## ANOVA Test for item_type ########################"
        )
        group_columns = ["participant_id", "trial_id", "item_type"]
        # Calculate percentages
        grouped_counts = df.groupby(group_columns).size().reset_index(name="counts")

        perform_nway_anova_with_lsd(
            grouped_counts, "counts", ["item_type"], perform_lsd=True
        )
# %%
