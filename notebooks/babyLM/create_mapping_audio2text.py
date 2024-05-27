import glob
import os
import time
import pandas as pd
import pickle

clean_audio_path = "/om/user/luwo/projects/data/peoples_speech/audio_files/clean"
dirty_audio_path = "/om/user/luwo/projects/data/peoples_speech/audio_files/dirty"
txt_path = "/om/user/luwo/projects/data/peoples_speech/alignments/data/SPAS/peoples-speech-joint-filtered_88M"
save_path = "/om/user/luwo/projects/data/peoples_speech/"

print("making cache")
start_t = time.time()
clean_audio_names_to_path = {
    "clean_" + x.split("/")[-1].split(".")[0]: x
    for x in glob.glob(f"{clean_audio_path}/*/*.flac")
}
print(f"Found {len(clean_audio_names_to_path)} clean audio files")

dirty_audio_names_to_path = {
    "dirty_" + x.split("/")[-1].split(".")[0]: x
    for x in glob.glob(f"{dirty_audio_path}/*/*.flac")
}
print(f"Found {len(dirty_audio_names_to_path)} dirty audio files")

txt_list = {
    x.split("/")[-1].replace("_transcript.txt", ""): x
    for x in glob.glob(f"{txt_path}/*/*.txt")
}
print(f"Found {len(txt_list)} text files")

print("done making cache in {}s".format(time.time() - start_t))

mapping = []
for txt_file in txt_list.keys():
    print(f"Checking {txt_file}")
    if txt_file in clean_audio_names_to_path.keys():
        mapping.append(
            (txt_list[txt_file], clean_audio_names_to_path[txt_file], "clean")
        )
    if txt_file in dirty_audio_names_to_path.keys():
        mapping.append(
            (txt_list[txt_file], dirty_audio_names_to_path[txt_file], "dirty")
        )

print(f"Found {len(mapping)} matches")

# pickle at save_path
with open(os.path.join(save_path, "mapping_88M.pkl"), "wb") as f:
    pickle.dump(mapping, f)


# store mapping as csv
df = pd.DataFrame(mapping, columns=["Text File", "Audio File", "Clean/Dirty"])
df.to_csv(os.path.join(save_path, "mapping_88M.csv"), index=False)
