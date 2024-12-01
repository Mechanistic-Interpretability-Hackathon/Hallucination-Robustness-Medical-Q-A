import tqdm
import time
import random
import threading
import concurrent.futures as futures
from functools import wraps
import pandas as pd
import numpy as np
pd.set_option('display.max_colwidth', None)
from ast import literal_eval
import re
import goodfire
import os
import sys
sys.path.append('..')
sys.path.append('medhalt')
from src.medhalt.medhalt.models.utils import PromptDataset
from src.med_llm_evaluation.medical_evaluator import MedicalLLMEvaluator
from dotenv import load_dotenv
load_dotenv()
GOODFIRE_API_KEY = os.getenv('GOODFIRE_API_KEY')
RATE_LIMIT = 99

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
        self.results = pd.DataFrame(columns=['id', 'feature_activation', 'prompt', 'question', 'options', 'correct_index', 'response', 'hallucinated', 'error'])
        self.feature_activation = 0

    def set_feature_activation(self, feature_name, activation_value):
        """Set feature activation for the variant"""
        self.feature_activation = activation_value
        self.variant.reset()
        selected_feature = self.client.features.search(feature_name, top_k=1)[0][0]
        self.variant.set(selected_feature, self.feature_activation, mode='pin')

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
            future_to_datapoint = {
                executor.submit(self.generate_variant_response, row['prompt']): row
                for row in dataset
            }
            print(len(self.results))

            # Process results as they complete
            for future in tqdm.tqdm(
                futures.as_completed(future_to_datapoint), 
                total=len(future_to_datapoint)
            ):
                datapoint = future_to_datapoint[future]
                try:
                    result = future.result()
                    hallucinated = self.process_response_for_hallucination(result)

                    # Create a new row for the DataFrame
                    new_row = pd.DataFrame([{
                        'id': datapoint['id'],
                        'feature_activation': self.feature_activation,
                        'prompt': datapoint['prompt'],
                        'question': datapoint['question'],
                        'options': datapoint['options'],
                        'correct_index': datapoint['correct_index'],
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
                        'id': datapoint['id'],
                        'feature_activation': self.feature_activation,
                        'prompt': datapoint['prompt'],
                        'question': datapoint['question'],
                        'options': datapoint['options'],
                        'correct_index': datapoint['correct_index'],
                        'response': None,
                        'error': str(e),
                        'hallucinated': None
                    }])
                    self.results = pd.concat([self.results, new_row], ignore_index=True)

            return self.results

def get_hallucination_rate(df: pd.DataFrame):
    """
    Calculate the hallucination rate for the results dataframe.
    """
    # Plot the hallucination rate for each feature activation.
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

    # Do a scatter plot of the hallucination rate for each feature activation. Sorting the rows of the dataset in terms of the feature activation.
    hallucination_rates_df = hallucination_rates_df.sort_values(by='feature_activation')
    hallucination_rates = hallucination_rates_df['hallucination_rate'].to_list()

    return hallucination_rates, error_bars


def main():
    # Load dataset
    base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    relevant_features = {
        0: "The model should not recommend technological or medical interventions",
        1: "Medical imaging techniques and procedures",
        2: "Medical case presentations with complex patient symptoms"
    }
    # User parameters
    selected_feature_index = 1
    filename = f'feature_{selected_feature_index}_results'
    min_activation, max_activation = -0.5, 0.5
    feature_activation_steps = 8
    n_examples = 200

    selected_feature_name = relevant_features[selected_feature_index]
    fct_ds = PromptDataset(dataset_name='FCT', prompt_template_fn=lambda x: x)
    sampled_fct_ds = fct_ds[:n_examples]  # Sample the first n rows

    # Initialize GoodFire client
    variant = goodfire.Variant(base_model_name)
    client = GoodFireClient(
        api_key=GOODFIRE_API_KEY,
        variant=variant,
        requests_per_minute=RATE_LIMIT
    )
    if selected_feature_name:
        medical_dataset_accuracy = []
        for feature_activation in np.linspace(min_activation, max_activation, feature_activation_steps):
            # Benchmark hallucination rate
            client.set_feature_activation(selected_feature_name, feature_activation)
            # Run generation and hallucination check
            _ = client.get_responses_for_dataset(sampled_fct_ds)

            # Benchmark general medical capabilities.
            evaluator = MedicalLLMEvaluator(client.client, client.variant)
            accuracy, _, _, _ = evaluator.run_evaluation(
                k=100,             # number of samples
                random_seed=42,    # for reproducibility
                max_workers=32,    # concurrent API calls
                subject_name=None  # optionally filter by subject
            )
            medical_dataset_accuracy.append(accuracy)

        # Save the results to a TSV file
        client.results.to_csv(f'data/{filename}.tsv', index=False, sep='\t')

        # Save 'cleaned out' version of the dataset, removing rows with errors on the generation or hallucination check
        results_cleaned = client.results.dropna(subset=['response', 'hallucinated'])
        results_cleaned.to_csv(f'data/{filename}_clean.tsv', index=False, sep='\t')

        # Calculate hallucination rates and save all the results into a TSV file
        hallucination_rates, error_bars = get_hallucination_rate(results_cleaned)
        results_df = pd.DataFrame({
            'feature_activation': np.linspace(min_activation, max_activation, feature_activation_steps),
            'hallucination_rate': hallucination_rates,
            'hallucination_rate_error': error_bars,
            'accuracy': medical_dataset_accuracy
        })
        results_df.to_csv(f'data/feature_{selected_feature_index}_benchmark_results.tsv', index=False, sep='\t')

if __name__ == "__main__":
    main()
