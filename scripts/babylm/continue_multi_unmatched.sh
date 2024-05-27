#!/bin/bash

#SBATCH --output=/om/user/luwo/projects/MIT_prosody/slurm_log/%j.out     # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=/om/user/luwo/projects/MIT_prosody/slurm_log/%j.err  # where to store error messages

#SBATCH -p evlab
#SBATCH -t 47:00:00 
#SBATCH -N 1                  # one node
#SBATCH --ntasks-per-node=1
#SBATCH -c 4                   # 4 virtual CPU cores
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G             # 40 GB of RAM

# # Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

echo "Running on node: $(hostname)"

# Binary or script to execute
python src/train.py experiment=babylm/whisbert_multi_unmatched ckpt_path=/om/user/luwo/projects/MIT_prosody/logs/train/runs/2023-08-08/18-18-31/checkpoints/last.ckpt

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0
