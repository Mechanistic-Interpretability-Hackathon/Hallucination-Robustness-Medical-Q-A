import argparse
import asyncio
import os
import pandas as pd
import numpy as np
import goodfire
from src.hallucination_llm_evaluation.utils import load_features
from src.medhalt.medhalt.models.utils import PromptDataset
from src.med_llm_evaluation.medical_evaluator import AsyncMedicalLLMEvaluator
from src.hallucination_llm_evaluation.hallucination_benchmark import AsyncGoodFireClient
from dotenv import load_dotenv
load_dotenv()

GOODFIRE_API_KEY = os.getenv('GOODFIRE_API_KEY')
RATE_LIMIT = 99
FEATURES_PATH = 'src/hallucination_llm_evaluation/relevant_features.json'
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

def get_hallucination_rate(df: pd.DataFrame):
    """Calculate the hallucination rate for the results dataframe."""
    # Plot the hallucination rate for each feature activation
    hallucination_rates = {}
    for feature_activation in df['feature_activation'].unique():
        feature_activation_df = df[df['feature_activation'] == feature_activation]
        hallucination_rate = len(feature_activation_df[feature_activation_df['hallucinated'] == True]) / len(feature_activation_df['hallucinated'].dropna())
        hallucination_rates[feature_activation] = hallucination_rate
    hallucination_rates_df = pd.DataFrame(hallucination_rates.items(), columns=['feature_activation', 'hallucination_rate'])

    # Get the hallucination counts for each feature activation
    ### First, create an empty dataframe with all possible feature activations
    all_feature_activations = df['feature_activation'].unique()
    hallucination_value_counts = df[df['hallucinated'] == True]['feature_activation'].value_counts().reindex(all_feature_activations, fill_value=0).sort_index()
    total_counts = df['feature_activation'].value_counts().sort_index()
    # Assign Poisson errors to each hallucination count
    poisson_errors = hallucination_value_counts.apply(lambda x: x**0.5)
    # Propagate the errors to the hallucination rate
    error_bars = list(poisson_errors.values / total_counts.values)

    # Sort the rows of the dataset by feature activation
    hallucination_rates_df = hallucination_rates_df.sort_values(by='feature_activation')
    hallucination_rates = hallucination_rates_df['hallucination_rate'].to_list()

    return hallucination_rates, error_bars

async def main(args):
    # Load dataset
    base_model_name = MODEL_NAME

    # Load features from JSON file
    relevant_features = load_features(FEATURES_PATH)
    
    # Set up filenames and parameters from command line args
    filename = f'feature_{args.selected_feature_index}_results'

    selected_feature = relevant_features[args.selected_feature_index]
    fct_ds = PromptDataset(dataset_name='FCT', prompt_template_fn=lambda x: x)
    sampled_fct_ds = fct_ds[:args.n_hallucination_examples]  # Sample the first n rows

    # Initialize GoodFire client
    variant = goodfire.Variant(base_model_name)
    client = AsyncGoodFireClient(
        api_key=GOODFIRE_API_KEY,
        variant=variant,
        requests_per_minute=RATE_LIMIT,
        batch_size=args.batch_size
    )

    if selected_feature:
        medical_dataset_accuracy = []
        for activation_value in np.linspace(args.min_activation, args.max_activation, args.feature_activation_steps):
            # Benchmark hallucination rate
            client.set_feature_activation(selected_feature, activation_value)
            # Run generation and hallucination check
            _ = await client.get_responses_for_dataset(sampled_fct_ds)

            # Benchmark general medical capabilities
            evaluator = AsyncMedicalLLMEvaluator(client.client, client.variant)
            accuracy, _, _, _ = await evaluator.run_evaluation(
                k=args.n_capabilities_examples,  # number of samples
                random_seed=42,                  # for reproducibility
                subject_name=None                # optionally filter by subject
            )
            medical_dataset_accuracy.append(accuracy)

            # Save the results to a TSV file
            client.results.to_csv(f'src/data/{filename}.tsv', index=False, sep='\t')

        # Save 'cleaned out' version of the dataset
        results_cleaned = client.results.dropna(subset=['response', 'hallucinated'])
        results_cleaned.to_csv(f'src/data/{filename}_clean.tsv', index=False, sep='\t')

        # Calculate hallucination rates and save results
        hallucination_rates, error_bars = get_hallucination_rate(results_cleaned)
        results_df = pd.DataFrame({
            'feature_activation': np.linspace(args.min_activation, args.max_activation, args.feature_activation_steps),
            'hallucination_rate': hallucination_rates,
            'hallucination_rate_error': error_bars,
            'accuracy': medical_dataset_accuracy
        })
        results_df.to_csv(f'src/data/feature_{args.selected_feature_index}_benchmark_results.tsv', index=False, sep='\t')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run hallucination analysis with feature steering.')
    parser.add_argument('--selected_feature_index', type=int, required=True,
                      help='Index of the selected feature.')
    parser.add_argument('--min_activation', type=float, default=-0.5,
                      help='Minimum activation value.')
    parser.add_argument('--max_activation', type=float, default=0.5,
                      help='Maximum activation value.')
    parser.add_argument('--feature_activation_steps', type=int, default=8,
                      help='Number of activation steps.')
    parser.add_argument('--n_hallucination_examples', type=int, default=200,
                      help='Number of examples to process for the hallucination rate calculation.')
    parser.add_argument('--n_capabilities_examples', type=int, default=200,
                      help='Number of examples to process for the medical Q&A capabilities precision calculation.')
    parser.add_argument('--batch_size', type=int, default=10,
                      help='Number of examples to process concurrently in each batch.')

    args = parser.parse_args()
    
    # Set up asyncio event loop with proper policy for Windows compatibility
    if os.name == 'nt':  # Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run the async main function
    asyncio.run(main(args))
