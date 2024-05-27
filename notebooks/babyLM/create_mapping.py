import os
import glob
import pandas as pd
import time
from tqdm import tqdm


def create_audio_text_mapping(text_root, clean_audio_root, dirty_audio_root, savepath):
    # Build audio cache
    print("Building audio cache")
    start_t = time.time()
    clean_audio_set = set(
        [
            "clean_" + os.path.splitext(os.path.basename(x))[0]
            for x in glob.glob(f"{clean_audio_root}/**/*.flac", recursive=True)
        ]
    )
    dirty_audio_set = set(
        [
            "dirty_" + os.path.splitext(os.path.basename(x))[0]
            for x in glob.glob(f"{dirty_audio_root}/**/*.flac", recursive=True)
        ]
    )
    print(f"Done building audio cache in {time.time() - start_t}s")

    # Get all text files
    text_files = glob.glob(f"{text_root}/**/*_transcript.txt", recursive=True)
    print(f"Found {len(text_files)} text files")

    mapping = []

    # Process each text file
    for text_file in tqdm(text_files):
        base_name = os.path.splitext(os.path.basename(text_file))[0].replace(
            "_transcript", ""
        )

        # Look for matching audio file in clean audio set
        if "clean_" + base_name in clean_audio_set:
            mapping.append([text_file, "clean_" + base_name])

        # Look for matching audio file in dirty audio set
        if "dirty_" + base_name in dirty_audio_set:
            mapping.append([text_file, "dirty_" + base_name])

    # Convert the mapping to a DataFrame and save as CSV
    df = pd.DataFrame(mapping, columns=["Text File", "Audio File"])
    df.to_csv(savepath, index=False)

    return df


# Replace with your actual paths
text_root = "/om/user/luwo/projects/data/peoples_speech/alignments/data/SPAS/peoples-speech-joint-100M"
clean_audio_root = "/om/user/luwo/projects/data/peoples_speech/audio_files/clean"
dirty_audio_root = "/om/user/luwo/projects/data/peoples_speech/audio_files/dirty"
savepath = f"/om/user/luwo/projects/data/peoples_speech/mapping_100M.csv"

os.makedirs(os.path.dirname(savepath), exist_ok=True)
df = create_audio_text_mapping(text_root, clean_audio_root, dirty_audio_root, savepath)
