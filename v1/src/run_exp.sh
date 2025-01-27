#!/bin/bash

#python3 train.py trainer.max_epochs=4 tags="[sam_masks]" data.train_batch_size=4 model=sam data=v1-sam modality_encoder=sam model.optimizer.lr=1e-5
#python3 train.py trainer.max_epochs=4 tags="[sam_masks]" data.train_batch_size=4 model=sam data=v1-sam modality_encoder=sam model.optimizer.lr=1e-6
python3 train.py trainer.max_epochs=4 tags="[sam_masks]" data.train_batch_size=1 model=sam data=v1-sam modality_encoder=sam model.optimizer.lr=1e-5
#python3 train.py trainer.max_epochs=4 tags="[sam_masks]" data.train_batch_size=2 model=sam data=v1-sam modality_encoder=sam model.optimizer.lr=1e-6
#python3 train.py model=default trainer.max_epochs=4 data.force_remake=False tags="[clip_points]" data.data_source="points" data.train_batch_size=32 model.optimizer.lr=1e-4
#python3 train.py trainer.max_epochs=4 data.force_remake=False data.debug=False data.train_batch_size=32 model=default model.optimizer.lr=1e-5
#python3 train.py trainer.max_epochs=4 data.force_remake=False data.debug=False data.train_batch_size=32 model=default model.optimizer.lr=1e-6
#python3 train.py trainer.max_epochs=4 data.force_remake=False data.debug=False data.train_batch_size=64 model=default model.optimizer.lr=1e-5
#python3 train.py trainer.max_epochs=4 data.force_remake=False data.debug=False data.train_batch_size=64 model=default model.optimizer.lr=1e-6
#python3 train.py trainer.max_epochs=4 data.force_remake=False data.debug=False data.train_batch_size=128 model=default model.optimizer.lr=1e-5
#python3 train.py trainer.max_epochs=4 data.force_remake=False data.debug=False data.train_batch_size=128 model=default model.optimizer.lr=1e-6
#python3 train.py model=sam data=v1-sam modality_encoder=sam trainer.max_epochs=4 data.train_batch_size=4 model.optimizer.lr=1e-4
#python3 train.py model=sam data=v1-sam modality_encoder=sam trainer.max_epochs=4 data.train_batch_size=2 model.optimizer.lr=1e-4
#python3 train.py model=default trainer.max_epochs=4 data.force_remake=False data.debug=False data.train_batch_size=32 model.optimizer.lr=1e-4
#python3 train.py model=default trainer.max_epochs=4 data.force_remake=False data.debug=False data.train_batch_size=64 model.optimizer.lr=1e-4
#python3 train.py model=default trainer.max_epochs=4 data.force_remake=False data.debug=False data.train_batch_size=128 model.optimizer.lr=1e-4
