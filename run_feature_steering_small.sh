#!/bin/bash

# Execute the command with the current parameters
python src/hallucination_llm_evaluation/feature_steering_hallucination.py \
  --selected_feature_index 0 \
  --min_activation -1 \
  --max_activation 1 \
  --feature_activation_steps 9 \
  --n_hallucination_examples 500 \
  --n_capabilities_examples 100 \
  --batch_size 100