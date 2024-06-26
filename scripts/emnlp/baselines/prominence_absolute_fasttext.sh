#!/bin/bash

#SBATCH --output=/om/user/luwo/projects/MIT_prosody/slurm_log/%j.out
#SBATCH --error=/om/user/luwo/projects/MIT_prosody/slurm_log/%j.err

#SBATCH -p evlab
#SBATCH -t 9:00:00 
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
# SBATCH --constraint=16GB 
#SBATCH --mem=15G

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

# Command to execute Python script
python3 src/models/baselines/train_gamma.py \
                        --NUM_RUNS 5 \
                        --INPUT_SIZE 300 \
                        --DEVICE cuda \
                        --OUTPUT_SIZE 2 \
                        --EPOCHS 15 \
                        --BATCH_SIZE 4096 \
                        --DATA_DIR "/om/user/luwo/projects/data/baselines/baseline_data/prominence_absolute" \
                        --GLOVE_PATH "/om/user/luwo/projects/data/models/glove/glove.6B.300d.txt" \
                        --FASTTEXT_PATH "/om/user/luwo/projects/data/models/fastText/cc.en.300.bin" \
                        --EMB_MODEL "fasttext" \

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0
