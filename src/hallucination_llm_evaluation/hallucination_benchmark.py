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
from src.medhalt.medhalt.models.utils import PromptDataset
from src.med_llm_evaluation.medical_evaluator import AsyncMedicalLLMEvaluator
from dotenv import load_dotenv
load_dotenv()

GOODFIRE_API_KEY = os.getenv('GOODFIRE_API_KEY')
RATE_LIMIT = 100
FEATURES_PATH = 'src/hallucination_llm_evaluation/relevant_features.json'
LOGS_FOLDER_PATH = 'src/hallucination_llm_evaluation/logs'
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

class AsyncRateLimiter:
    """Async rate limiter using a semaphore and sliding window"""
    def __init__(self):
        self.semaphore = asyncio.Semaphore(RATE_LIMIT)
        self.request_times = []
        self.window_size = 60  # 1 minute window
        self.requests_per_minute = RATE_LIMIT
        self.lock = asyncio.Lock()
        self.logger = logging.getLogger('async_processor.rate_limiter')

    async def acquire(self):
        """Acquire rate limit token"""
        start_time = time.time()
        current_time = start_time
        
        self.logger.debug(f"Attempting to acquire rate limit token. Current queue size: {len(self.request_times)}")
        
        async with self.lock:
            # Remove timestamps older than our window
            initial_queue = len(self.request_times)
            self.request_times = [t for t in self.request_times 
                                if current_time - t < self.window_size]
            self.logger.debug(f"Cleaned queue from {initial_queue} to {len(self.request_times)} requests")
            
            # If we're at capacity, wait until we have room
            if len(self.request_times) >= self.requests_per_minute:
                self.logger.warning(f"Rate limit reached. Queue size: {len(self.request_times)}")
                
            while len(self.request_times) >= self.requests_per_minute:
                wait_time = self.request_times[0] + self.window_size - current_time
                if wait_time > 0:
                    self.logger.info(f"Rate limit sleep for {wait_time:.2f} seconds")
                    await asyncio.sleep(wait_time)
                current_time = time.time()
                self.request_times = [t for t in self.request_times 
                                    if current_time - t < self.window_size]
            
            self.request_times.append(current_time)
            await self.semaphore.acquire()
            
        total_time = time.time() - start_time
        self.logger.debug(f"Token acquired after {total_time:.2f} seconds")

    async def release(self):
        """Release rate limit token"""
        self.logger.debug("Releasing rate limit token")
        self.semaphore.release()

class AsyncGoodFireClient:
    def __init__(self, api_key: str, variant: goodfire.Variant, batch_size=100):
        self.client = goodfire.AsyncClient(api_key)
        self.variant = variant
        self.rate_limiter = AsyncRateLimiter()
        self.batch_size = batch_size
        self.feature_activation = 0
        self.logger = logging.getLogger('async_processor.goodfire_client')
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
        start_time = time.time()
        
        for retry in range(max_retries):
            try:
                self.logger.debug(f"Attempting request (retry {retry})")
                await self.rate_limiter.acquire()
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
                finally:
                    await self.rate_limiter.release()
            except Exception as e:
                self.logger.error(f"Request failed: {str(e)}")
                if retry == max_retries - 1:
                    raise e
                wait_time = initial_wait * (2 ** retry) + random.uniform(0, 1)
                self.logger.warning(f"Retrying in {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        total_time = time.time() - start_time
        self.logger.error(f"Max retries exceeded after {total_time:.2f} seconds")
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
        batch_start = time.time()
        self.logger.info(f"Starting batch processing of {len(batch)} items")

        async def process_row(row):
            row_start = time.time()
            try:
                self.logger.debug(f"Processing row {row['id']}")
                result = await self.generate_variant_response(row['prompt'])
                hallucinated = self.process_response_for_hallucination(result)
                row_time = time.time() - row_start
                self.logger.debug(f"Row {row['id']} completed in {row_time:.2f} seconds")
                return {
                    'id': row['id'],
                    'feature_activation': self.feature_activation,
                    'prompt': row['prompt'],
                    'question': row['question'],
                    'options': row['options'],
                    'correct_index': row['correct_index'],
                    'response': result,
                    'error': None,
                    'hallucinated': hallucinated,
                }
            except Exception as e:
                row_time = time.time() - row_start
                self.logger.error(f"Error processing row {row['id']}: {str(e)}")
                return {
                    'id': row['id'],
                    'feature_activation': self.feature_activation,
                    'prompt': row['prompt'],
                    'question': row['question'],
                    'options': row['options'],
                    'correct_index': row['correct_index'],
                    'response': None,
                    'error': str(e),
                    'hallucinated': None,
                }

        tasks = [process_row(row) for row in batch]
        results = await asyncio.gather(*tasks)
        batch_time = time.time() - batch_start
        avg_time_per_row = batch_time / len(batch)
        self.logger.info(f"Batch completed in {batch_time:.2f} seconds. Average time per row: {avg_time_per_row:.2f} seconds")
        return pd.DataFrame(results)

    async def get_responses_for_dataset(self, dataset, filename):
        """Process dataset with concurrent batches"""
        total_start = time.time()
        self.logger.info(f"Starting processing of {len(dataset)} total items in batches of {self.batch_size}")
        
        results = []
        for i in tqdm.tqdm(range(0, len(dataset), self.batch_size)):
            batch = dataset[i:i + self.batch_size]
            batch_results = await self.process_batch(batch)
            results.append(batch_results)
            self.results = pd.concat(results, ignore_index=True)
            self.results.to_csv(f'src/data/{filename}.tsv', index=False, sep='\t')
            
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
