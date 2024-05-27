import os
import shutil
import random


def copy_random_dirs(src_dir, dst_dir, num_dirs=5):
    # Get a list of all subdirectories in src_dir
    subdirs = [
        os.path.join(src_dir, d)
        for d in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, d))
    ]

    # Randomly select num_dirs directories
    dirs_to_copy = random.sample(subdirs, num_dirs)

    # Ensure the destination directory exists
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # Copy each directory
    for dir_path in dirs_to_copy:
        shutil.copytree(dir_path, os.path.join(dst_dir, os.path.basename(dir_path)))


# Use the function
copy_random_dirs(
    "/om/user/luwo/projects/data/peoples_speech/audio_files/dirty",
    "/om/user/luwo/projects/MIT_prosody/notebooks/babyLM/random_sample",
    num_dirs=20,
)
