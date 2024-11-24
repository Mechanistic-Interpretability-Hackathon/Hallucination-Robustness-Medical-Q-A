"""
Example usage of the MedicalLLMEvaluator.

This script demonstrates how to:
1. Set up the evaluator with your API credentials
2. Run an evaluation on medical questions
3. Access and interpret the results

REQUIRES:
- GOODFIRE_API_KEY environment variable
- HF_TOKEN environment variable (huggingface, for downloading datasets)

"""

import os
import goodfire
import logging
from src.med_llm_evaluation.medical_evaluator import MedicalLLMEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API key from environment variable
api_key = os.getenv('GOODFIRE_API_KEY')
if not api_key:
    raise ValueError("Please set the GOODFIRE_API_KEY environment variable")

# Initialize Goodfire client
client = goodfire.Client(api_key)
variant = goodfire.Variant("meta-llama/Meta-Llama-3-8B-Instruct")

# Initialize evaluator
evaluator = MedicalLLMEvaluator(client, variant)

# Run evaluation
accuracy, kappa, results, stats = evaluator.run_evaluation(
    k=30,              # number of samples
    random_seed=42,    # for reproducibility
    max_workers=10,    # concurrent API calls
    subject_name=None  # optionally filter by subject
)

# Results are already logged by the evaluator
# You can also access them programmatically:
print(f"\nAccuracy: {accuracy:.3f}")
print(f"Kappa: {kappa:.3f}")

# Show some example results
print("\nSample of results:")
print(results[['prompt', 'true_answer', 'predicted_answer', 'correct']].head())

# Access statistical analysis results
print("\nStatistical significance:", "Yes" if stats['better_than_random'] else "No")
print(f"Effect size: {stats['effect_size']:.3f} ({stats['effect_size_interpretation']})")
