import pandas as pd
import os
import sys
import argparse
from utils import (
    print_mean,
    split_decimal_small,
    perform_gee,
    pairwise_comparisons_with_bonferroni,
    analyze_reaction_times,
    print_statistics,
    plot_reaction_time_histograms_valid,
    plot_line_chart,
    preprocess_df,
    multiple_choice_accuracy_calc,
    perform_nway_anova_with_lsd,
    fix_condition,
    perform_kruskal_wallis_with_posthoc,
)


def main(exp_num, method):
    directory = os.path.join("results", f"exp{exp_num}")
    output_path = os.path.join(directory, "output.log")
    theo_path = f"exp{exp_num[0]}_theo.xlsx"
    thet_path = f"exp{exp_num[0]}_thet.xlsx"

    pd.set_option("display.max_columns", None)
    thet_df = pd.read_excel(thet_path)
    theo_df = pd.read_excel(theo_path)

    # Redirect output to a file
    os.makedirs(directory, exist_ok=True)
    original_stdout = sys.stdout
    try:
        with open(output_path, "w") as f:
            sys.stdout = f
            # 0.1
            print(
                "############################################# Statistics 0.1 #############################################"
            )
            print_mean(thet_df, "Thetikoi Statistics")
            print_mean(theo_df, "Theoritikoi Statistics")
            # 1.1
            df = preprocess_df(theo_df)
            df["item_type"] = df["item_type"].apply(fix_condition)
            # 2b
            if exp_num == "2b":
                df = multiple_choice_accuracy_calc(df)

            decimal_df, small_df = split_decimal_small(df)

            for idx, df in enumerate([decimal_df, small_df]):
                title = "Decimal" if idx == 0 else "Small"
                print(
                    f"\n\n############################################# {title}s #############################################"
                )
                # 1.2.1
                results_dict, bounds, filtered_df = analyze_reaction_times(
                    df, method=method
                )
                print(
                    "\n\n############################################# Anova Test 1.2.2 #############################################"
                )
                print(filtered_df.value_counts("item_type"))

                perform_kruskal_wallis_with_posthoc(filtered_df, "accuracy", ["item_type"])
                perform_kruskal_wallis_with_posthoc(filtered_df, "reaction_time", ["item_type"])

                print_statistics(results_dict, method)
                plot_reaction_time_histograms_valid(
                    df,
                    results_dict,
                    bounds,
                    method=method,
                    title=title,
                    directory=directory,
                )

                # 1.3
                comparisons = [
                    ("C(item_type)[T.IC]", "Intercept"),
                    ("C(item_type)[T.II1]", "Intercept"),
                    ("C(item_type)[T.II2]", "Intercept"),
                    ("C(item_type)[T.II1]", "C(item_type)[T.IC]"),
                    ("C(item_type)[T.II2]", "C(item_type)[T.IC]"),
                    ("C(item_type)[T.II2]", "C(item_type)[T.II1]"),
                ]
                if exp_num == "2b":
                    # keep the original
                    filtered_df_accuracy = df
                else:
                    filtered_df_accuracy = filtered_df
                # 1.3.1
                print(
                    "\n\n############################################# Statistics Accuracy 1.3.1 #############################################"
                )
                gee_results = perform_gee(
                    filtered_df_accuracy, "accuracy", "item_type", analysis_type="logistic"
                )
                results = pairwise_comparisons_with_bonferroni(gee_results, comparisons)
                print(pd.DataFrame(results).round(3))

                # 1.3.2
                print(
                    "\n\n############################################# Statistics Reaction Time 1.3.2 #############################################"
                )
                gee_results = perform_gee(
                    filtered_df, "reaction_time", "item_type", analysis_type="linear"
                )
                results = pairwise_comparisons_with_bonferroni(gee_results, comparisons)
                print(pd.DataFrame(results).round(3))

                plot_line_chart(
                    filtered_df_accuracy,
                    title=title,
                    metric="accuracy",
                    directory=directory,
                    metric_name="Accuracy",
                )
                plot_line_chart(
                    filtered_df,
                    title=title,
                    metric="reaction_time",
                    method="mean",
                    directory=directory,
                    metric_name="Reaction Time",
                )

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        sys.stdout = original_stdout
        print(f"All output has been saved to {output_path}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run analysis for experiment.")
    parser.add_argument("--exp_num", type=str, required=True, help="Experiment number")
    parser.add_argument(
        "--method",
        type=str,
        choices=["mean", "median"],
        required=True,
        help="Method to use",
    )
    args = parser.parse_args()

    # Pass arguments to main function
    main(exp_num=args.exp_num, method=args.method)
