#!/bin/bash

python3 train.py trainer.max_epochs=5 data.force_remake=True data.debug=True
