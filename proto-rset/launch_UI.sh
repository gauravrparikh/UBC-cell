#!/usr/bin/env bash
#SBATCH --job-name=webserver # Job name
#SBATCH --mail-type=NONE          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/web_server.out
#SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=10
#SBATCH --mem=160gb                     # Job memory request
#SBATCH --time=168:00:00               # Time limit hrs:min:sec
#SBATCH --partition=rudin
#SBATCH --gres=gpu:a6000:1
#SBATCH --account=rudin
#SBATCH --nodelist=rudin-01

python3 -u app.py