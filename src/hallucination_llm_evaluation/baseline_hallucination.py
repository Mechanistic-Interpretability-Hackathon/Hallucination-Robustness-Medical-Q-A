import argparse
import asyncio
import os
import pandas as pd
import numpy as np
import goodfire
from src.hallucination_llm_evaluation.utils import load_features
from src.medhalt.medhalt.models.utils import PromptDataset
from src.med_llm_evaluation.medical_evaluator import AsyncMedicalLLMEvaluator
from src.hallucination_llm_evaluation.hallucination_benchmark import AsyncGoodFireClient, GOODFIRE_API_KEY, RATE_LIMIT, MODEL_NAME
from dotenv import load_dotenv
load_dotenv()

async def main(args):
    # Load dataset
    base_model_name = MODEL_NAME
    
    # Set up filenames and parameters from command line args
    filename = 'baseline_results'
    fct_ds = PromptDataset(dataset_name='FCT', prompt_template_fn=lambda x: x)
    sampled_fct_ds = fct_ds[:args.n_examples]  # Sample the first n rows

    # Initialize GoodFire client
    variant = goodfire.Variant(base_model_name)
    client = AsyncGoodFireClient(
        api_key=GOODFIRE_API_KEY,
        variant=variant,
        batch_size=args.batch_size
    )

    # Benchmark hallucination rate without feature steering
    # Run generation and hallucination check, and save results
    _ = await client.get_responses_for_dataset(sampled_fct_ds, filename=filename)

    # Save 'cleaned out' version of the dataset
    results_cleaned = client.results.dropna(subset=['response', 'hallucinated'])
    results_cleaned.to_csv(f'src/data/{filename}_clean.tsv', index=False, sep='\t')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run hallucination analysis with feature steering.')
    parser.add_argument('--n_examples', type=int, default=200,
                      help='Number of examples to process for the hallucination rate calculation.')
    parser.add_argument('--batch_size', type=int, default=RATE_LIMIT,
                      help='Number of examples to process concurrently in each batch.')

    args = parser.parse_args()
    
    # Set up asyncio event loop with proper policy for Windows compatibility
    if os.name == 'nt':  # Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run the async main function
    asyncio.run(main(args))