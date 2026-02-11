#!/bin/bash

python task_4.py --embed_dim 50  --embed_type glove --epochs 5 --mode train
python task_4.py --embed_dim 100  --embed_type glove --epochs 5 --mode train
python task_4.py --embed_dim 200  --embed_type glove --epochs 5 --mode train
python task_4.py --embed_dim 300  --embed_type glove --epochs 5 --mode train

python task_4.py --embed_dim 50  --embed_type glove --epochs 5 --mode test
python task_4.py --embed_dim 100  --embed_type glove --epochs 5 --mode test
python task_4.py --embed_dim 200  --embed_type glove --epochs 5 --mode test
python task_4.py --embed_dim 300  --embed_type glove --epochs 5 --mode test

python task_4.py --embed_dim 50  --embed_type svd --epochs 5 --mode train
python task_4.py --embed_dim 100  --embed_type svd --epochs 5 --mode train
python task_4.py --embed_dim 200  --embed_type svd --epochs 5 --mode train
python task_4.py --embed_dim 300  --embed_type svd --epochs 5 --mode train

python task_4.py --embed_dim 50  --embed_type svd --epochs 5 --mode test
python task_4.py --embed_dim 100  --embed_type svd --epochs 5 --mode test
python task_4.py --embed_dim 200  --embed_type svd --epochs 5 --mode test
python task_4.py --embed_dim 300  --embed_type svd --epochs 5 --mode test
