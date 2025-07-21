#!/usr/bin/env bash
#SBATCH --job-name=test_samp # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/test_sampling_rset_%j.out
#SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=10
#SBATCH --mem=160gb                     # Job memory request
#SBATCH --time=96:00:00               # Time limit hrs:min:sec
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1

export CUB200_DIR=/usr/xtmp/lam135/datasets/CUB_200_2011_2/
python3 -u test_sampling_rset.py