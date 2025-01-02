# %%
import os
import pandas as pd

# Define the directory containing the data files
data_directory = "./data"
final_merged_a_b = []  # List to hold all DataFrames from case a and b
final_merged_c = []  # List to hold all DataFrames from case c

# Process each .txt file in the directory
for file in os.listdir(data_directory):
    if file.endswith(".txt"):
        txt_file_path = os.path.join(data_directory, file)
        xlsx_file_path = os.path.join(data_directory, file.replace(".txt", ".xlsx"))

        # Read the .txt file
        txt_df = pd.read_csv(txt_file_path, sep="\t")
        columns = [
            "Session_Name_",
            "ACCURACY",
            "RT",
            "KEY_PRESSED",
            "correctresponse",
            "identifier",
            "block",
            "practicestatus",
        ]

        # Check if the corresponding .xlsx file exists
        if os.path.exists(xlsx_file_path):
            # Read the .xlsx file
            xlsx_df = pd.read_excel(xlsx_file_path)

            # Case c: If "identifier" exists in the xlsx file
            if "identifier" in xlsx_df.columns:
                txt_df = txt_df[columns]
                merged_df = pd.merge(txt_df, xlsx_df, on="identifier", how="inner")
                final_merged_c.append(merged_df)
            else:
                # Case b: Concatenate txt_df and xlsx_df
                concatenated_df = pd.concat(
                    [txt_df.reset_index(drop=True), xlsx_df.reset_index(drop=True)],
                    axis=1,
                )
                final_merged_a_b.append(concatenated_df)
        else:
            columns += ["condition", "categoryofoperands"]
            txt_df = txt_df[columns]
            if (
                str(txt_df["Session_Name_"].iloc[0]).startswith("10")
                and str(txt_df["Session_Name_"].iloc[0]) != "107"
            ):

                final_merged_a_b.append(txt_df)
            else:
                final_merged_c.append(txt_df)

# Combine and save results for case a and b
if final_merged_a_b:
    combined_df_a_b = pd.concat(final_merged_a_b, axis=0)
    combined_df_a_b = combined_df_a_b[
        combined_df_a_b["practicestatus"] != "practice"
    ].reset_index(drop=True)

    output_file_a_b = os.path.join("final_merged_thetikoi.xlsx")
    combined_df_a_b.to_excel(output_file_a_b, index=False)

    # Print summary statistics for case a and b
    if "RT" in combined_df_a_b.columns:
        print(f'Mean RT: {combined_df_a_b["RT"].mean() / 1000:.2f} seconds')
    if "ACCURACY" in combined_df_a_b.columns:
        print(f'Mean ACCURACY: {combined_df_a_b["ACCURACY"].mean():.2f} percent')

# Combine and save results for case c
if final_merged_c:
    combined_df_c = pd.concat(final_merged_c, axis=0).reset_index(drop=True)
    combined_df_c = combined_df_c[
        combined_df_c["practicestatus"] != "practice"
    ].reset_index(drop=True)

    output_file_c = os.path.join("final_merged_theoritikoi.xlsx")
    combined_df_c.to_excel(output_file_c, index=False)

print("Processing complete. The resulting files have been saved in the data directory.")
