import shutil
import pandas as pd
import os


def copy_files_from_mapping(mapping_file: str, dst: str):
    # Read mapping file
    df = pd.read_csv(mapping_file)

    # Create the destination directory if it doesn't exist
    os.makedirs(dst, exist_ok=True)

    # Iterate over the rows of the DataFrame
    for _, row in df.iterrows():
        # Copy audio file to new location
        shutil.copy(row["audio"], dst)

        # Copy transcript file to new location
        shutil.copy(row["transcript"], dst)


# Use the function
copy_files_from_mapping("./file_mapping.csv", "./final_sample")
