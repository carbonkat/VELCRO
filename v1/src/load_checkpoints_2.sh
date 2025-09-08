#!/bin/bash

#python3 train.py model=default trainer.max_epochs=10 data.force_remake=False tags="[clip_masks]" data.data_source="masks" data.train_batch_size=8 ckpt_path="./logs/train/runs/2025-01-24_19-37-54/checkpoints/epoch_003.ckpt"
#python3 train.py model=default trainer.max_epochs=20 data.force_remake=False tags="[clip_masks]" data.data_source="masks" data.train_batch_size=32 ckpt_path="./logs/train/runs/2025-01-24_19-43-02/checkpoints/epoch_003.ckpt"
python3 train.py model=default trainer.max_epochs=20 tags="[clip_points]" data.data_source="points" data.train_batch_size=16 ckpt_path="./logs/train/runs/2025-01-26_09-41-35/checkpoints/epoch_003.ckpt"
python3 train.py model=default trainer.max_epochs=20 tags="[clip_points]" data.data_source="points" data.train_batch_size=32 ckpt_path="./logs/train/runs/2025-01-25_01-52-18/checkpoints/epoch_019.ckpt"
python3 train.py model=default trainer.max_epochs=20 tags="[clip_points]" data.data_source="points" data.train_batch_size=64 ckpt_path="./logs/train/runs/2025-01-26_08-02-37/checkpoints/epoch_003.ckpt"
python3 train.py model=default trainer.max_epochs=20 tags="[clip_points]" data.data_source="points" data.train_batch_size=64 ckpt_path="./logs/train/runs/2025-01-26_09-41-35/checkpoints/epoch_003.ckpt"
python3 train.py model=default trainer.max_epochs=20 tags="[clip_points]" data.data_source="points" data.train_batch_size=128 ckpt_path="./logs/train/runs/2025-01-26_06-17-01/checkpoints/epoch_003.ckpt"
