import pandas as pd
from src.data.components.datasets import (
    PeoplesMultiModalDataset,
    PeoplesMultiModalPackagedDataset,
)


def main():
    # Load the file mapping
    print(f"Starting to prepare the packs")
    mapping_path = (
        "/om/user/luwo/projects/data/peoples_speech/mappings/mapping_100M.csv"
    )
    save_dir = "/om/user/luwo/projects/data/peoples_speech/packs/100M_fast"  # directory to save the sample packs

    mapping = pd.read_csv(mapping_path)
    alignment_files = mapping.iloc[:, 0].tolist()
    flac_files = mapping.iloc[:, 1].tolist()

    # Initialize the parameters
    sr = 16000  # sample rate
    num_words_per_sample = 50  # number of words per sample
    dataset_total_words = int(100e6)  # total words in the dataset
    samples_per_pack = 10000  # number of samples per pack

    # Instantiate the dataset
    dataset = PeoplesMultiModalDataset(
        alignment_files=alignment_files,
        flac_files=flac_files,
        sr=sr,
        num_words_per_sample=num_words_per_sample,
        dataset_total_words=dataset_total_words,
    )

    # Wrap the dataset with the packaged dataset class
    packaged_dataset = PeoplesMultiModalPackagedDataset(
        dataset=dataset,
        num_files_per_pack=samples_per_pack,
        save_dir=save_dir,
    )

    # Prepare the packs
    packaged_dataset.prepare_packs()

    print(f"Finished preparing the packs")


if __name__ == "__main__":
    main()
