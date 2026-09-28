#!/bin/bash
#SBATCH --job-name=run
#SBATCH --partition=Standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus=2g.45gb:1
#SBATCH --output=logs/gpu_job_%j.log
#SBATCH --time=0-6:00:00

curl --location-trusted -u 22b1029:f07b3c72a55d9598cc17fa115061476c "https://internet-sso.iitb.ac.in/login.php"

python run3.py --selection_method "norm" --max_heads 10
python run3.py --selection_method "norm" --max_heads 20
python run3.py --selection_method "norm" --max_heads 30

python run3.py --selection_method "rank" --max_heads 10
python run3.py --selection_method "rank" --max_heads 20
python run3.py --selection_method "rank" --max_heads 30