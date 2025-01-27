#!/bin/bash

#python3 train.py trainer.max_epochs=4 data.force_remake=False tags="[clip_masks]" data.data_source="masks" data.train_batch_size=8 model=default model.optimizer.lr=1e-5
#python3 train.py trainer.max_epochs=4 data.force_remake=False tags="[clip_masks]" data.data_source="masks" data.train_batch_size=8 model=default model.optimizer.lr=1e-6
#python3 train.py trainer.max_epochs=4 data.force_remake=False tags="[clip_masks]" data.data_source="masks" data.train_batch_size=16 model=default model.optimizer.lr=1e-5
#python3 train.py trainer.max_epochs=1 data.force_remake=False tags="[clip_masks]" data.data_source="masks" data.train_batch_size=16 model=default model.optimizer.lr=1e-6
#python3 train.py trainer.max_epochs=4 data.force_remake=False tags="[clip_masks]" data.data_source="masks" data.train_batch_size=32 model=default model.optimizer.lr=1e-5
#python3 train.py trainer.max_epochs=4 data.force_remake=False tags="[clip_masks]" data.data_source="masks" data.train_batch_size=32 model=default model.optimizer.lr=1e-6
#python3 train.py trainer.max_epochs=4 data.force_remake=False tags="[clip_masks]" data.data_source="masks" data.train_batch_size=64 model=default model.optimizer.lr=1e-5
#python3 train.py trainer.max_epochs=4 data.force_remake=False tags="[clip_masks]" data.data_source="masks" data.train_batch_size=64 model=default model.optimizer.lr=1e-6
#python3 train.py trainer.max_epochs=4 data.force_remake=False tags="[clip_masks]" data.data_source="masks" data.train_batch_size=128 model=default model.optimizer.lr=1e-5
#python3 train.py trainer.max_epochs=4 data.force_remake=False tags="[clip_masks]" data.data_source="masks" data.train_batch_size=128 model=default model.optimizer.lr=1e-6
#python3 train.py model=default trainer.max_epochs=4 data.force_remake=False tags="[clip_masks]" data.data_source="masks" data.train_batch_size=8 model.optimizer.lr=1e-4
#python3 train.py model=default trainer.max_epochs=4 data.force_remake=False tags="[clip_masks]" data.data_source="masks" data.train_batch_size=16 model.optimizer.lr=1e-4
python3 train.py model=default trainer.max_epochs=4 data.force_remake=False tags="[clip_points]" data.data_source="points" data.train_batch_size=32 model.optimizer.lr=1e-4
python3 train.py model=default trainer.max_epochs=4 data.force_remake=False tags="[clip_points]" data.data_source="points" data.train_batch_size=64 model.optimizer.lr=1e-4
python3 train.py model=default trainer.max_epochs=4 data.force_remake=False tags="[clip_points]" data.data_source="points" data.train_batch_size=128 model.optimizer.lr=1e-4
