#!/bin/bash

TOKENIZERS_PARALLELISM=false python3 train.py logger=wandb trainer.max_epochs=5 data.train_batch_size=24
