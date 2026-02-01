#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=Standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus=2g.45gb:1
#SBATCH --output=logs/jupyter.log
#SBATCH --error=logs/jupyter.err
#SBATCH --time=0-12:00:00

JUPYTER_PORT=8180
COMPUTE_NODE=$(hostname -s)
echo "ssh -N -f -R ${JUPYTER_PORT}:${COMPUTE_NODE}:${JUPYTER_PORT} ${USER}@localhost"
echo "Jupyter Lab will run on: ${COMPUTE_NODE}"

unset HF_HUB_CACHE

jupyter lab --no-browser --ip=0.0.0.0 --port=${JUPYTER_PORT}