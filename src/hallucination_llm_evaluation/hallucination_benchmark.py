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
# Add the mech-interp directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.hallucination_llm_evaluation.utils import load_features
from src.medhalt.medhalt.models.utils import PromptDataset
from src.med_llm_evaluation.medical_evaluator import AsyncMedicalLLMEvaluator
from dotenv import load_dotenv
load_dotenv()

GOODFIRE_API_KEY = os.getenv('GOODFIRE_API_KEY')
RATE_LIMIT = 99
FEATURES_PATH = 'src/hallucination_llm_evaluation/relevant_features.json'
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

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
