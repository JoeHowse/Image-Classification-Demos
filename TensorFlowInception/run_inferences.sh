#!/usr/bin/env bash

# Activate the TensorFlow virtual environment.
source ~/SDKs/TensorFlow/TensorFlow/bin/activate

# Set TensorFlow's logging level.
export TF_CPP_MIN_LOG_LEVEL=2

# Run the Python script.
python run_inferences.py
