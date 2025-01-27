#!/bin/bash

python3 train.py model=default trainer.max_epochs=10 tags="[clip_points]" data.data_source="points" data.train_batch_size=8 ckpt_path="./logs/train/runs/2025-01-26_06-14-25/checkpoints/epoch_003.ckpt"
python3 train.py model=default trainer.max_epochs=20 tags="[clip_points]" data.data_source="points" data.train_batch_size=32 ckpt_path="./logs/train/runs/2025-01-26_06-15-00/checkpoints/epoch_003.ckpt"
python3 train.py model=default trainer.max_epochs=20 tags="[clip_points]" data.data_source="points" data.train_batch_size=16 ckpt_path="./logs/train/runs/2025-01-26_09-22-38/checkpoints/epoch_003.ckpt"
python3 train.py model=default trainer.max_epochs=20 tags="[clip_points]" data.data_source="points" data.train_batch_size=128 ckpt_path="./logs/train/runs/2025-01-26_10-17-56/checkpoints/epoch_003.ckpt"
#python3 train.py model=default trainer.max_epochs=20 data.force_remake=False tags="[clip_masks]" data.data_source="masks" data.train_batch_size=32 ckpt_path="./logs/train/runs/2025-01-24_21-32-40/checkpoints/epoch_003.ckpt"
#python3 train.py model=default trainer.max_epochs=20 data.force_remake=False tags="[clip_masks]" data.data_source="masks" data.train_batch_size=64 ckpt_path="./logs/train/runs/2025-01-24_21-30-29/checkpoints/epoch_003.ckpt"
#python3 train.py model=default trainer.max_epochs=20 data.force_remake=False tags="[clip_masks]" data.data_source="masks" data.train_batch_size=64 ckpt_path="./logs/train/runs/2025-01-24_23-13-30/checkpoints/epoch_003.ckpt"
#python3 train.py model=default trainer.max_epochs=20 data.force_remake=False tags="[clip_masks]" data.data_source="masks" data.train_batch_size=128 ckpt_path="./logs/train/runs/2025-01-25_00-02-23/checkpoints/epoch_003.ckpt"
