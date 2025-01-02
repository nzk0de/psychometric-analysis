import os
import pandas as pd
from eyelinkio import read_edf
from tqdm import tqdm  # Import tqdm for progress visualization

# Specify the root directory containing participant folders
root_dir = "data/eye_tracker"

# Initialize an empty list to store fixation data
all_fixations = []

# Loop through all subdirectories in the root directory with tqdm
for folder_name in tqdm(os.listdir(root_dir), desc="Processing Participants"):
    folder_path = os.path.join(root_dir, folder_name)

    # Check if the path is a directory
    if os.path.isdir(folder_path):
        # Search for .edf file in the current folder
        edf_files = [f for f in os.listdir(folder_path) if f.endswith(".edf")]
        assert len(edf_files) <= 1, f"Multiple .edf files found in {folder_path}"

        if edf_files:
            edf_file = edf_files[0]
            edf_path = os.path.join(folder_path, edf_file)

            try:
                # Read the EDF file
                edf_data = read_edf(edf_path)
                df_dict = edf_data.to_pandas()

                # Extract fixation data and messages
                if "discrete" in df_dict and "fixations" in df_dict["discrete"]:
                    fixations = df_dict["discrete"]["fixations"]
                    messages = df_dict["discrete"]["messages"]

                    # Filter messages for PREPARE_SEQUENCE
                    filtered_messages = messages[
                        messages["msg"].str.contains("PREPARE_SEQUENCE", na=False)
                    ]
                    filtered_messages.reset_index(inplace=True)
                    filtered_messages.drop(columns=["index"], inplace=True)

                    # Calculate end times for each message
                    filtered_messages["etime"] = (
                        filtered_messages["stime"]
                        .shift(-1)
                        .fillna(filtered_messages["stime"] + 50)
                    )

                    # Initialize trial_id column
                    fixations["trial_id"] = None

                    # Assign trial IDs to fixations
                    for idx, trial in filtered_messages.iterrows():
                        mask = (fixations["stime"] >= trial["stime"]) & (
                            fixations["etime"] <= trial["etime"]
                        )
                        fixations.loc[mask, "trial_id"] = idx + 1

                    # Add participant ID (folder name) to the fixation data
                    fixations["participant_id"] = folder_name

                    # Append to the list
                    all_fixations.append(fixations)

            except Exception as e:
                print(f"Error processing {edf_path}: {e}")
                continue


# Concatenate all fixation data into a single DataFrame
if all_fixations:
    final_fixation_df = pd.concat(all_fixations, ignore_index=True)
    print("Final fixation DataFrame created successfully.")
    print(final_fixation_df.head())
else:
    print("No fixation data found.")

# Save the final DataFrame to a CSV file
final_fixation_df.to_excel("all_fixations.xlsx", index=False)

# %%
df = pd.read_excel("all_fixations.xlsx")

# %%
