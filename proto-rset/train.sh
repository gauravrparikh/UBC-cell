#!/usr/bin/env bash
#SBATCH --job-name=multimod # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jonathan.donnelly@maine.edu     # Where to send mail
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/naive_train_%j.out
#SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=5
#SBATCH --mem=90gb                     # Job memory request
#SBATCH --time=12:00:00               # Time limit hrs:min:sec
#SBATCH --partition=compsci-gpu
#not SBATCH --account=rudin
#SBATCH --gres=gpu:1

python3 -m protopnet train-vanilla-cos --dataset=ubc_cells --backbone=identity # --dataset-dir=/usr/xtmp/lam135/datasets/CUB_200_2011_2/