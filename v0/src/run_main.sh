#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --mem=48GB
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --output=JOB-%j.log
#SBATCH -e JOB-%j.err
#SBATCH --mail-type=ALL
#SBATCH --account=robin_group
#SBATCH --mail-user=l2hebert@uwaterloo.ca

srun python main.py --clip_model=$1
