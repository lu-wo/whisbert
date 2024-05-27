#!/bin/bash

#SBATCH --output=/nese/mit/group/evlab/u/luwo/projects/MIT_prosody/slurm_log/%j.out     # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=/nese/mit/group/evlab/u/luwo/projects/MIT_prosody/slurm_log/%j.err  # where to store error messages

#SBATCH -p evlab
#SBATCH -t 9:00:00 
#SBATCH -N 1                  # one node
#SBATCH -c 2                   # 4 virtual CPU cores
#SBATCH --gres=gpu:1
#SBATCH --constraint=20GB 
#SBATCH --mem=15G             # 40 GB of RAM


# # Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

echo "Running on node: $(hostname)"

# Binary or script to execute
python src/train.py experiment=emnlp/finetuning/duration_regression_abs_bert

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0
