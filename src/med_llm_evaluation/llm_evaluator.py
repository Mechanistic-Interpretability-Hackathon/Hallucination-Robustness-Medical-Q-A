import re
import random
import asyncio
from typing import Dict, List, Optional, Tuple
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
import logging

logger = logging.getLogger(__name__)

class AsyncLLMEvaluator:
    """Handles async evaluation of LLM performance on medical questions."""
    
    def __init__(self, client, variant, batch_size: int = 10):
        """
        Initialize evaluator with API client and variant.
        
        Args:
            client: The async API client instance
            variant: The model variant to use
            batch_size: Number of concurrent requests to process
        """
        self.variant = variant
        self.client = client
        self.batch_size = batch_size
    
    async def evaluate(self, 
                X: List[str], 
                y: List[int], 
                k: int,
                random_seed: Optional[int] = None) -> Tuple[float, float, pd.DataFrame]:
        """
        Evaluate model performance on a sample of questions.
        
        Args:
            X (List[str]): List of prompts
            y (List[int]): List of correct answers
            k (int): Number of samples to evaluate
            random_seed (Optional[int]): Random seed for reproducibility
            
        Returns:
            Tuple[float, float, pd.DataFrame]: Accuracy, kappa, and detailed results
        """
        X_sample, y_sample, indices = self._sample_data(X, y, k, random_seed)
        return await self._evaluate_concurrent(X_sample, y_sample)
    
    def _sample_data(self, 
                    X: List[str], 
                    y: List[int], 
                    k: int, 
                    random_seed: Optional[int] = None) -> Tuple[List[str], List[int], List[int]]:
        """Randomly sample k items from data."""
        if random_seed is not None:
            random.seed(random_seed)
        
        n = len(X)
        indices = random.sample(range(n), k)
        X_sampled = [X[i] for i in indices]
        y_sampled = [y[i] for i in indices]
        
        return X_sampled, y_sampled, indices
    
    async def _evaluate_concurrent(self, 
                           X_sample: List[str], 
                           y_sample: List[int]) -> Tuple[float, float, pd.DataFrame]:
        """Evaluate samples concurrently in batches."""
        y_pred = []
        
        # Process in batches
        for i in range(0, len(X_sample), self.batch_size):
            batch_prompts = X_sample[i:i + self.batch_size]
            
            # Create tasks for batch
            tasks = [
                self._get_model_response(prompt)
                for prompt in batch_prompts
            ]
            
            # Run batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Error in API call: {result}")
                    y_pred.append(-1)
                else:
                    y_pred.append(result if result is not None else -1)
        
        return self._calculate_metrics(X_sample, y_sample, y_pred)
    
    async def _get_model_response(self, prompt: str) -> Optional[int]:
        """Get integer response from model."""
        try:
            response = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.variant,
                stream=False,
                max_completion_tokens=50
            )
            content = response.choices[0].message['content']
            return self._extract_first_valid_integer(content)
        except Exception as e:
            logger.error(f"Error getting model response: {e}")
            return None
    
    @staticmethod
    def _extract_first_valid_integer(text: str) -> int:
        """Extract first valid integer from text."""
        pattern = r'\d+'
        match = re.search(pattern, text)
        
        if match:
            number = int(match.group())
            if number in [0, 1, 2, 3]:
                return number
        return -1
    
    def _calculate_metrics(self, 
                         X_sample: List[str], 
                         y_sample: List[int],
                         y_pred: List[int]) -> Tuple[float, float, pd.DataFrame]:
        """Calculate evaluation metrics."""
        valid_indices = [i for i, pred in enumerate(y_pred) if pred != -1]
        y_pred_valid = [y_pred[i] for i in valid_indices]
        y_sample_valid = [y_sample[i] for i in valid_indices]
        
        if len(valid_indices) > 0:
            accuracy = accuracy_score(y_sample_valid, y_pred_valid)
            kappa = cohen_kappa_score(y_sample_valid, y_pred_valid)
            conf_matrix = confusion_matrix(y_sample_valid, y_pred_valid, labels=[0,1,2,3])
            per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        else:
            accuracy = 0
            kappa = 0
            per_class_accuracy = [0, 0, 0, 0]
        
        results_df = pd.DataFrame({
            'prompt': X_sample,
            'true_answer': y_sample,
            'predicted_answer': y_pred,
            'correct': [pred == true for pred, true in zip(y_pred, y_sample)]
        })
        
        # Log results
        logger.info(f"\nEvaluation Results:")
        logger.info(f"Total samples: {len(X_sample)}")
        logger.info(f"Valid predictions: {len(valid_indices)}")
        logger.info(f"Accuracy: {accuracy:.3f}")
        logger.info(f"Cohen's Kappa: {kappa:.3f}")
        logger.info("\nPer-class accuracy:")
        for i, acc in enumerate(per_class_accuracy):
            logger.info(f"Class {i}: {acc:.3f}")
        
        return accuracy, kappa, results_df