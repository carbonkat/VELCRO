#!/bin/bash
export PROJECT_ROOT='.'
TOKENIZERS_PARALLELISM=false python3 train.py logger=wandb trainer.max_epochs=5 data.train_batch_size=32 model.loss.learnable_temperature=False model.loss.temperature=0.2
