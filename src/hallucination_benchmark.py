import tqdm
import time
import random
import threading
import concurrent.futures as futures
from functools import wraps
import pandas as pd
pd.set_option('display.max_colwidth', None)
from ast import literal_eval
import re
import goodfire
import os
import sys
sys.path.append('..')
sys.path.append('medhalt')
from src.medhalt.medhalt.models.utils import PromptDataset
from dotenv import load_dotenv
load_dotenv()
GOODFIRE_API_KEY = os.getenv('GOODFIRE_API_KEY')

def exponential_backoff(max_retries=5, initial_wait=1, max_wait=60):
    """
    Decorator that implements exponential backoff for rate-limited functions.
    
    Args:
        max_retries (int): Maximum number of retry attempts
        initial_wait (float): Initial wait time in seconds
        max_wait (float): Maximum wait time in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "rate limit" in str(e).lower() or retries < max_retries:
                        wait_time = min(
                            initial_wait * (2 ** retries) + random.uniform(0, 1),
                            max_wait
                        )
                        time.sleep(wait_time)
                        retries += 1
                        if retries <= max_retries:
                            continue
                    raise e
            return None
        return wrapper
    return decorator

class RateLimitedExecutor:
    """
    Thread pool executor with rate limiting capabilities.
    """
    def __init__(self, max_workers=32, requests_per_minute=100):
        self.max_workers = max_workers
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = time.time()
        self.lock = threading.Lock()

    def wait_for_rate_limit(self):
        """Ensure minimum interval between requests"""
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_request_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_request_time = time.time()

class GoodFireClient:
    def __init__(self, api_key, variant, requests_per_minute=100):
        self.client = goodfire.Client(api_key)
        self.variant = variant
        self.executor = RateLimitedExecutor(
            max_workers=32,
            requests_per_minute=requests_per_minute
        )
        # Initialize DataFrame with correct columns
        self.results = pd.DataFrame(columns=['prompt', 'response', 'error', 'hallucinated'])

    @exponential_backoff(max_retries=5, initial_wait=1, max_wait=60)
    def generate_variant_response(self, prompt):
        """
        Generate a response to a prompt using the llama model with rate limiting
        """
        self.executor.wait_for_rate_limit()
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.variant,
            temperature=0,  # Set temperature to 0 to disable sampling
        )
        return response.choices[0].message['content']

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

    def get_responses_for_dataset(self, dataset):
        """
        Get responses for a dataset using thread pool executor with rate limiting
        """
        with futures.ThreadPoolExecutor(max_workers=self.executor.max_workers) as executor:
            # Submit all tasks to the executor
            future_to_prompt = {
                executor.submit(self.generate_variant_response, row['prompt']): row['prompt']
                for row in dataset
            }
            
            # Process results as they complete
            for future in tqdm.tqdm(
                futures.as_completed(future_to_prompt), 
                total=len(future_to_prompt)
            ):
                prompt = future_to_prompt[future]
                try:
                    result = future.result()
                    hallucinated = self.process_response_for_hallucination(result)
                    
                    # Create a new row for the DataFrame
                    new_row = pd.DataFrame([{
                        'prompt': prompt,
                        'response': result,
                        'error': None,
                        'hallucinated': hallucinated
                    }])
                    
                    # Concatenate the new row to the existing DataFrame
                    self.results = pd.concat([self.results, new_row], ignore_index=True)
                    
                except Exception as e:
                    print(f"Error processing prompt: {str(e)}")
                    # Add error case to DataFrame
                    new_row = pd.DataFrame([{
                        'prompt': prompt,
                        'response': None,
                        'error': str(e),
                        'hallucinated': None
                    }])
                    self.results = pd.concat([self.results, new_row], ignore_index=True)
            
            return self.results


def main():
    # Load dataset
    dataset_name = "FCT"
    fct_ds = PromptDataset(dataset_name=dataset_name, prompt_template_fn=lambda x: x)
    # Sample the first 10000 rows
    sampled_fct_ds = fct_ds[500:5000]

    # Initialize GoodFire client
    variant = goodfire.Variant("meta-llama/Meta-Llama-3-8B-Instruct")

    client = GoodFireClient(
        api_key=GOODFIRE_API_KEY,
        variant=variant,
        requests_per_minute=99
    )

    responses = client.get_responses_for_dataset(sampled_fct_ds)
    responses.to_csv('fct_responses.tsv', index=False, sep='\t')

if __name__ == "__main__":
    main()