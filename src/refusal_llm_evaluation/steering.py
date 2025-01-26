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
import os
import sys
import logging
from datetime import datetime
# Add the mech-interp directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.hallucination_llm_evaluation.utils import load_features
from dotenv import load_dotenv
load_dotenv()

GOODFIRE_API_KEY = os.getenv('GOODFIRE_API_KEY')
RATE_LIMIT = 200
FEATURES_PATH = 'src/refusal_llm_evaluation/refusal_features.json'
LOGS_FOLDER_PATH = 'src/refusal_llm_evaluation/logs'
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{LOGS_FOLDER_PATH}/async_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('async_processor')

class RefusalSteeringExperiment:
    def __init__(self, api_key: str, variant: goodfire.Variant, batch_size=100):
        self.client = goodfire.AsyncClient(api_key)
        self.variant = variant
        self.batch_size = batch_size
        self.feature_activation = 0
        self.logger = logging.getLogger('async_processor.goodfire_client')
        self.results = pd.DataFrame(columns=[
            'id', 'feature_activation', 'entity', 'system_prompt', 'user_prompt', 'known', 'response', 'refused', 'error'
        ])

    def set_feature_activation(self, feature: goodfire.Feature, activation_value: float):
        """Set feature activation for the variant"""
        self.feature_activation = activation_value
        self.variant.reset()
        self.variant.set(feature, self.feature_activation)

    async def generate_variant_response(self, prompt):
        """Generate a response with rate limiting"""
        try:
            api_start = time.time()
            response = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.variant,
                temperature=0
            )
            api_time = time.time() - api_start
            self.logger.info(f"API request completed in {api_time:.2f} seconds")
            return response.choices[0].message['content']
        except Exception as e:
            self.logger.error(f"Request failed: {str(e)}")
            return None

    def process_response_for_refusal(self, response_string: str):
        """Process response to determine if it refused to answer"""
        return False
        

    async def process_batch(self, batch):
        """Process a batch of rows concurrently"""
        batch_start = time.time()
        self.logger.info(f"Starting batch processing of {len(batch)} items")

        async def process_row(row):
            row_start = time.time()
            self.logger.debug(f"Processing row")
            response = await self.generate_variant_response(row['prompt'])
            if response:
                refused = self.process_response_for_refusal(response)
            else:
                refused = None
            row_time = time.time() - row_start
            self.logger.debug(f"Row completed in {row_time:.2f} seconds")
            return {
                'feature_activation': self.feature_activation,
                'entity': row['entity'],
                'system_prompt': row['system_prompt'],
                'user_prompt': row['user_prompt'],
                'known': row['known'],
                'response': response,
                'refused': refused,
                'error': response is None
            }
        tasks = [process_row(row) for row in batch]
        results = await asyncio.gather(*tasks)
        batch_time = time.time() - batch_start
        avg_time_per_row = batch_time / len(batch)
        self.logger.info(f"Batch completed in {batch_time:.2f} seconds. Average time per row: {avg_time_per_row:.2f} seconds")
        return pd.DataFrame(results)

    async def get_responses_for_dataset(self, dataset: pd.DataFrame, filename: str):
        """Process dataset with concurrent batches"""
        total_start = time.time()
        self.logger.info(f"Starting processing of {len(dataset)} total items in batches of {self.batch_size}")
        
        results = []
        for i in tqdm.tqdm(range(0, len(dataset), self.batch_size)):
            batch = dataset[i:i + self.batch_size]
            batch_results = await self.process_batch(batch)
            results.append(batch_results)
            self.results = pd.concat(results, ignore_index=True)
            self.results.to_csv(f'src/refusal_llm_evaluation/{filename}.tsv', index=False, sep='\t')
            
            # Log progress statistics
            processed = min((i + self.batch_size), len(dataset))
            elapsed = time.time() - total_start
            rate = processed / elapsed
            remaining = (len(dataset) - processed) / rate if rate > 0 else 0
            
            self.logger.info(
                f"Progress: {processed}/{len(dataset)} items. "
                f"Rate: {rate:.2f} items/sec. "
                f"Est. remaining time: {remaining/60:.1f} minutes"
            )
            
        total_time = time.time() - total_start
        self.logger.info(f"Dataset processing completed in {total_time:.2f} seconds")
        return self.results


async def run_experiment(args):
    # Load dataset
    base_model_name = MODEL_NAME

    # Load features from JSON file
    relevant_features = load_features(FEATURES_PATH)

    # Set up filenames and parameters from command line args
    filename = f'feature_{args.selected_feature_index}_results'

    selected_feature = relevant_features[args.selected_feature_index]
    refusal_dataset = pd.read_csv('src/refusal_llm_evaluation/refusal_dataset.csv')

    # Initialize GoodFire client
    variant = goodfire.Variant(base_model_name)
    client = RefusalSteeringExperiment(
        api_key=GOODFIRE_API_KEY,
        variant=variant,
        batch_size=args.batch_size
    )

    if selected_feature:
        for activation_value in np.linspace(args.min_activation, args.max_activation, args.feature_activation_steps):
            # Benchmark hallucination rate
            client.set_feature_activation(selected_feature, activation_value)
            # Run generation and refusal check
            _ = await client.get_responses_for_dataset(refusal_dataset, filename=filename)

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
    parser.add_argument('--batch_size', type=int, default=10,
                      help='Number of examples to process concurrently in each batch.')

    args = parser.parse_args()
    
    # Run the async main function
    asyncio.run(run_experiment(args))