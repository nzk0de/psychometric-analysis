import pandas as pd
import os
import re


# Define directories
xlsx_dir = "data/eye_tracker_res/"
txt_dir = "data/eye_tracker/"

theo_dfs = []
thet_dfs = []
# Loop through each number in the directory
for filename in os.listdir(xlsx_dir):
    full_filename = os.path.join(xlsx_dir, filename)
    number = re.search(r"(\d+)\.xlsx", filename).group(1)
    # Construct the file paths for .xlsx and .txt
    txt_file = os.path.join(txt_dir, number, "RESULTS_FILE.txt")

    # Check if both files exist
    if os.path.exists(full_filename) and os.path.exists(txt_file):
        # Read the .xlsx file
        xlsx_data = pd.read_excel(full_filename)

        # Read the .txt file as a CSV, skipping rows where condition == 'practice'
        txt_data = pd.read_csv(txt_file, sep="\t")
        txt_data = txt_data.iloc[3:].reset_index(drop=True)  # Skip first 3 rows

        # Ensure that the number of rows matches before concatenation
        if len(xlsx_data) == len(txt_data):
            # Concatenate by row, adding columns from xlsx to txt, filling missing columns with NaN
            combined_data = pd.concat([txt_data, xlsx_data], axis=1, join="outer")
            if str(number).startswith("10") and not str(number) == "107":
                thet_dfs.append(combined_data)
            else:
                # Append the combined data to the list
                theo_dfs.append(combined_data)

        else:
            print(
                f"Warning: Row count mismatch for {full_filename} between .xlsx and .txt files."
            )
    else:
        print(f"Warning: Missing file for number {number}")


greek_to_latin = {"α": "a", "β": "b", "γ": "c"}

# Apply the conversion to the specified columns

for idx, df_list in enumerate([theo_dfs, thet_dfs]):
    # Concatenate all the data from each folder into a single DataFrame
    final_df = pd.concat(df_list, ignore_index=True)
    final_df[["correctmultiplechoice", "multipleresponse"]] = final_df[
        ["correctmultiplechoice", "multipleresponse"]
    ].applymap(lambda x: greek_to_latin.get(x, x))

    final_df = final_df.loc[:, ~final_df.columns.str.contains("^Unnamed")]

    # Clean 'Session_Name_' column: remove non-numeric values and convert to int
    final_df["Session_Name_"] = (
        final_df["Session_Name_"]
        .apply(lambda x: "".join(filter(str.isdigit, str(x))))
        .astype(int)
    )

    # Save the final combined DataFrame to a CSV file
    if idx:
        final_df.to_excel("exp2_thet.xlsx", index=False)
    else:
        final_df.to_excel("exp2_theo.xlsx", index=False)

print("Data combined and saved successfully.")
