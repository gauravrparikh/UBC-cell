#!/bin/bash
#SBATCH --job-name=DimensionReductionRun
#SBATCH --output=logs/output_%j.out
#SBATCH --error=logs/error_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=512G
#SBATCH --time=24:00:00

# --- Configuration ---
# Description: This script runs the Python-based embedding experiment.
# Adjust the variables in this section to configure the run.

# Path to the single-cell dataset file
DATA_PATH="/usr/xtmp/UBC2025/data/human_cell_atlas.h5ad"

# Base directory where all results (plots, embeddings) will be saved.
# A subdirectory will be created for each run (e.g., ./experiment_results/run_0_seed_42)
OUTPUT_DIR="./experiment_results"

# Number of cells to randomly sample from the dataset for each run 2480955
SAMPLE_SIZE=2480955

# The column in the .obs dataframe that contains the cell labels for coloring the plots
LABEL_COLUMN="cell_type"

# Number of times to repeat the full experiment with a different random seed
N_REPEATS=5

# --- Boolean Flags ---
# Set to 'true' to enable, or 'false' to disable.

# Enable to log-transform the data and select for highly variable genes before embedding.
DO_PREPROCESS=false

# Enable to skip re-computing an embedding if the output file already exists.
USE_EXISTING_FILES=false


# --- Script Execution ---

# Create the logs directory if it doesn't exist
mkdir -p logs

# Activate the virtual environment
echo "Activating virtual environment..."
source /usr/xtmp/UBC2025/.venv/bin/activate
echo "Environment activated."

# Prepare flags for the python script
# This allows us to conditionally add flags to the command
PREPROCESS_FLAG=""
if [ "$DO_PREPROCESS" = true ]; then
    PREPROCESS_FLAG="--preprocess"
fi

USE_EXISTING_FLAG=""
if [ "$USE_EXISTING_FILES" = true ]; then
    USE_EXISTING_FLAG="--use_existing"
fi

# Define the path to the python script from the Canvas
PYTHON_SCRIPT_PATH="/usr/xtmp/UBC2025/experiment_0.py" # Make sure to save the canvas code to this path

# Run the experiment
echo "Starting experiment script..."
uv run python ${PYTHON_SCRIPT_PATH} \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --sample_size ${SAMPLE_SIZE} \
    --label_column "${LABEL_COLUMN}" \
    --n_repeats ${N_REPEATS} \
    ${PREPROCESS_FLAG} \
    ${USE_EXISTING_FLAG}

echo "Experiment script finished."