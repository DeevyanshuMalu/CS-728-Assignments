#!/bin/bash
#SBATCH --job-name=run
#SBATCH --partition=Standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus=2g.45gb:1
#SBATCH --output=logs/gpu_job_%j.log
#SBATCH --time=0-6:00:00

# # Task 1: Memorization
# # RNN (tanh), no clipping
# python train.py --task mem --model rnn --alpha 0.0 \--clipstyle nothing \--nhid 50 --lr 0.01 --bs 20 --min_length 50 --max_length 200 \--maxiters 50000 --ebs 10000 --cbs 1000 --checkFreq 20 \--seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05 \--name A1_mem_rnn_tanh_noclip

# # RNN (tanh), with clipping (moderate)
# python train.py --task mem --model rnn --alpha 0.0 \--clipstyle rescale --cutoff 0.05 \--nhid 50 --lr 0.01 --bs 20 --min_length 50 --max_length 200 \--maxiters 50000 --ebs 10000 --cbs 1000 --checkFreq 20 \--seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05 \--name A2_mem_rnn_tanh_clip005

# # RNN (tanh), with clipping (aggressive)
# python train.py --task mem --model rnn --alpha 0.0 \--clipstyle rescale --cutoff 0.01 \--nhid 50 --lr 0.01 --bs 20 --min_length 50 --max_length 200 \--maxiters 50000 --ebs 10000 --cbs 1000 --checkFreq 20 \--seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05 \--name A3_mem_rnn_tanh_clip001

# GRU, no clipping (with gate diagnostics)
python train.py --task mem --model gru --alpha 0.0 \--clipstyle nothing --diagGates \--nhid 50 --lr 0.01 --bs 20 --min_length 50 --max_length 200 \--maxiters 50000 --ebs 10000 --cbs 1000 --checkFreq 20 \--seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05 \--name A4_mem_gru_noclip

# # GRU, with clipping (moderate, with gate diagnostics)
# python train.py --task mem --model gru --alpha 0.0 \--clipstyle rescale --cutoff 0.05 --diagGates \--nhid 50 --lr 0.01 --bs 20 --min_length 50 --max_length 200 \--maxiters 50000 --ebs 10000 --cbs 1000 --checkFreq 20 \--seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05 \--name A5_mem_gru_clip005

# # Task 2:  Multiplication (regression)
# # RNN (tanh), no clipping
# python train.py --task mul --model rnn --alpha 0.0 \--clipstyle nothing \--nhid 50 --lr 0.01 --bs 20 --min_length 50 --max_length 200 \--maxiters 50000 --ebs 10000 --cbs 1000 --checkFreq 20 \--seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05 \--name B1_mul_rnn_tanh_noclip

# # GRU, no clipping (with gate diagnostics)
# python train.py --task mul --model gru --alpha 0.0 \--clipstyle nothing --diagGates \--nhid 50 --lr 0.01 --bs 20 --min_length 50 --max_length 200 \--maxiters 50000 --ebs 10000 --cbs 1000 --checkFreq 20 \--seed 52 --valid_seed 12345 --collectDiags --diagBins 60 --satThresh 0.05 \--name B2_mul_gru_noclip