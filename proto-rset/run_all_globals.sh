#!/usr/bin/env bash
#SBATCH --job-name=test_samp # Job name
#SBATCH --mail-type=NONE          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/all_globals_%j_%a.out
#SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=10
#SBATCH --mem=160gb                     # Job memory request
#SBATCH --time=96:00:00               # Time limit hrs:min:sec
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --array=0

export CUB200_DIR=/usr/xtmp/lam135/datasets/CUB_200_2011_2/
export DOGS_DIR=/usr/xtmp/jcd97/datasets/stanford_dogs/
export CARS_DIR=/usr/xtmp/jcd97/datasets/cars/
python3 -u run_all_globals.py $SLURM_ARRAY_TASK_ID