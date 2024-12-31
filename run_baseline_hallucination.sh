#!/bin/bash

# Set the PYTHONPATH to the current directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Execute the command with the current parameters
python src/hallucination_llm_evaluation/baseline_hallucination.py \
  --n_examples 900 \
  --batch_size 100