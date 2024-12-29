import asyncio
import tqdm
import time
import random
from functools import wraps
import pandas as pd
import numpy as np
from ast import literal_eval
import re
import goodfire
import os
import argparse
from src.hallucination_llm_evaluation.utils import load_features
from src.medhalt.medhalt.models.utils import PromptDataset
from src.med_llm_evaluation.medical_evaluator import AsyncMedicalLLMEvaluator
from dotenv import load_dotenv
load_dotenv()

GOODFIRE_API_KEY = os.getenv('GOODFIRE_API_KEY')
RATE_LIMIT = 99

class AsyncRateLimiter:
    """Async rate limiter using a semaphore and sliding window"""
    def __init__(self, requests_per_minute: int):
        self.semaphore = asyncio.Semaphore(requests_per_minute)
        self.request_times = []
        self.window_size = 60  # 1 minute window
        self.requests_per_minute = requests_per_minute
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire rate limit token"""
        current_time = time.time()
        
        async with self.lock:
            # Remove timestamps older than our window
            self.request_times = [t for t in self.request_times 
                                if current_time - t < self.window_size]
            
            # If we're at capacity, wait until we have room
            while len(self.request_times) >= self.requests_per_minute:
                wait_time = self.request_times[0] + self.window_size - current_time
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                current_time = time.time()
                self.request_times = [t for t in self.request_times 
                                    if current_time - t < self.window_size]
            
            self.request_times.append(current_time)
            await self.semaphore.acquire()

    async def release(self):
        """Release rate limit token"""
        self.semaphore.release()

class AsyncGoodFireClient:
    def __init__(self, api_key: str, variant: goodfire.Variant, requests_per_minute=100, batch_size=10):
        self.client = goodfire.AsyncClient(api_key)
        self.variant = variant
        self.rate_limiter = AsyncRateLimiter(requests_per_minute)
        self.batch_size = batch_size
        self.feature_activation = 0
        # Initialize DataFrame with correct columns
        self.results = pd.DataFrame(columns=[
            'id', 'feature_activation', 'prompt', 'question', 'options',
            'correct_index', 'response', 'hallucinated', 'error'
        ])

    def set_feature_activation(self, feature: goodfire.Feature, activation_value: float):
        """Set feature activation for the variant"""
        self.feature_activation = activation_value
        self.variant.reset()
        self.variant.set(feature, self.feature_activation)

    async def generate_variant_response(self, prompt):
        """Generate a response with rate limiting"""
        max_retries = 5
        initial_wait = 1
        
        for retry in range(max_retries):
            try:
                await self.rate_limiter.acquire()
                try:
                    response = await self.client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.variant,
                        temperature=0
                    )
                    return response.choices[0].message['content']
                finally:
                    await self.rate_limiter.release()
            except Exception as e:
                if retry == max_retries - 1:
                    raise e
                wait_time = initial_wait * (2 ** retry) + random.uniform(0, 1)
                await asyncio.sleep(wait_time)
        raise Exception("Max retries exceeded")

    def process_response_for_hallucination(self, response_string):
        """Process response to determine if it contains hallucination"""
        try:
            # Option 1: Try to parse the response using ast.literal_eval
            response_dict = literal_eval(response_string)
            decision = response_dict.get('is_answer_correct', '').lower()
            if decision not in ['yes', 'no']:
                return None
            return decision == 'yes'
        except Exception:
            try:
                # Option 2: Try regex if ast.literal_eval fails
                pattern = r"['\"]is_answer_correct['\"]\s*:\s*['\"](\w+)['\"]"
                match = re.search(pattern, response_string)

                if match:
                    decision = match.group(1).lower()
                    if decision not in ['yes', 'no']:
                        return None
                    return decision == 'yes'
                else:
                    print("Could not find is_answer_correct in response using either method")
                    return None
            except Exception as regex_error:
                print(f"Both parsing methods failed. Error: {regex_error}")
                return None

    async def process_batch(self, batch):
        """Process a batch of rows concurrently"""
        async def process_row(row):
            try:
                result = await self.generate_variant_response(row['prompt'])
                hallucinated = self.process_response_for_hallucination(result)
                return {
                    'id': row['id'],
                    'feature_activation': self.feature_activation,
                    'prompt': row['prompt'],
                    'question': row['question'],
                    'options': row['options'],
                    'correct_index': row['correct_index'],
                    'response': result,
                    'error': None,
                    'hallucinated': hallucinated
                }
            except Exception as e:
                return {
                    'id': row['id'],
                    'feature_activation': self.feature_activation,
                    'prompt': row['prompt'],
                    'question': row['question'],
                    'options': row['options'],
                    'correct_index': row['correct_index'],
                    'response': None,
                    'error': str(e),
                    'hallucinated': None
                }

        tasks = [process_row(row) for row in batch]
        results = await asyncio.gather(*tasks)
        return pd.DataFrame(results)

    async def get_responses_for_dataset(self, dataset):
        """Process dataset with concurrent batches"""
        results = []
        for i in tqdm.tqdm(range(0, len(dataset), self.batch_size)):
            batch = dataset[i:i + self.batch_size]
            batch_results = await self.process_batch(batch)
            results.append(batch_results)
        
        self.results = pd.concat([self.results] + results, ignore_index=True)
        return self.results

def get_hallucination_rate(df: pd.DataFrame):
    """Calculate the hallucination rate for the results dataframe."""
    # Plot the hallucination rate for each feature activation
    hallucination_rates = {}
    for feature_activation in df['feature_activation'].unique():
        feature_activation_df = df[df['feature_activation'] == feature_activation]
        hallucination_rate = len(feature_activation_df[feature_activation_df['hallucinated'] == True]) / len(feature_activation_df['hallucinated'].dropna())
        hallucination_rates[feature_activation] = hallucination_rate
    hallucination_rates_df = pd.DataFrame(hallucination_rates.items(), columns=['feature_activation', 'hallucination_rate'])

    # Get the error bars
    hallucination_value_counts = df[df['hallucinated'] == True]['feature_activation'].value_counts().sort_index()
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
    base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    # Load features from JSON file
    relevant_features = load_features('relevant_features.json')
    
    # Set up filenames and parameters from command line args
    filename = f'feature_{args.selected_feature_index}_results'

    selected_feature = relevant_features[args.selected_feature_index]
    fct_ds = PromptDataset(dataset_name='FCT', prompt_template_fn=lambda x: x)
    sampled_fct_ds = fct_ds[:args.n_examples]  # Sample the first n rows

    # Initialize GoodFire client
    variant = goodfire.Variant(base_model_name)
    client = AsyncGoodFireClient(
        api_key=GOODFIRE_API_KEY,
        variant=variant,
        requests_per_minute=RATE_LIMIT
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
                k=100,             # number of samples
                random_seed=42,    # for reproducibility
                max_workers=32,    # concurrent API calls
                subject_name=None  # optionally filter by subject
            )
            medical_dataset_accuracy.append(accuracy)

        # Save the results to a TSV file
        client.results.to_csv(f'data/{filename}.tsv', index=False, sep='\t')

        # Save 'cleaned out' version of the dataset
        results_cleaned = client.results.dropna(subset=['response', 'hallucinated'])
        results_cleaned.to_csv(f'data/{filename}_clean.tsv', index=False, sep='\t')

        # Calculate hallucination rates and save results
        hallucination_rates, error_bars = get_hallucination_rate(results_cleaned)
        results_df = pd.DataFrame({
            'feature_activation': np.linspace(args.min_activation, args.max_activation, args.feature_activation_steps),
            'hallucination_rate': hallucination_rates,
            'hallucination_rate_error': error_bars,
            'accuracy': medical_dataset_accuracy
        })
        results_df.to_csv(f'data/feature_{args.selected_feature_index}_benchmark_results.tsv', index=False, sep='\t')

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
    parser.add_argument('--n_examples', type=int, default=200,
                      help='Number of examples to process.')
    parser.add_argument('--batch_size', type=int, default=10,
                      help='Number of examples to process concurrently in each batch.')

    args = parser.parse_args()
    
    # Set up asyncio event loop with proper policy for Windows compatibility
    if os.name == 'nt':  # Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run the async main function
    asyncio.run(main(args))
