from pathlib import Path
import json
import ast
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

class DataHandler:
    """Handles loading and preprocessing of medical evaluation data."""
    
    def __init__(self, cache_dir: str = ".cache/med_eval"):
        """
        Initialize DataHandler with cache directory.
        
        Args:
            cache_dir (str): Directory for caching downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_path = self.cache_dir / "med_halt_data.json"
        
    def load_data(self) -> Tuple[List[str], List[int], List[str]]:
        """
        Load data, using cache if available, otherwise download fresh.
        
        Returns:
            Tuple[List[str], List[int], List[str]]: Processed prompts, labels, and subject names
        """
        if self.data_path.exists():
            logger.info("Loading data from cache...")
            return self._load_from_cache()
        
        logger.info("Downloading fresh data...")
        return self._download_and_cache()
    
    def _load_from_cache(self) -> Tuple[List[str], List[int], List[str]]:
        """Load processed data from cache."""
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            return data['prompts'], data['labels'], data['subject_names']
        except Exception as e:
            logger.error(f"Error loading cached data: {e}")
            logger.info("Falling back to fresh download...")
            return self._download_and_cache()
    
    def _download_and_cache(self) -> Tuple[List[str], List[int], List[str]]:
        """Download fresh data, process it, and cache the results."""
        try:
            dataset = load_dataset("openlifescienceai/Med-HALT", "reasoning_FCT")
            train_data = dataset['train']
            
            prompts, labels, subject_names = self._process_data(train_data)
            
            # Cache the processed data
            cache_data = {
                'prompts': prompts,
                'labels': labels,
                'subject_names': subject_names
            }
            with open(self.data_path, 'w') as f:
                json.dump(cache_data, f)
            
            return prompts, labels, subject_names
                
        except Exception as e:
            logger.error(f"Error downloading/processing data: {e}")
            raise
    
    def _process_data(self, dataset) -> Tuple[List[str], List[int], List[str]]:
        """Process raw dataset into prompts, labels, and subject names."""
        prompts = []
        labels = []
        subject_names = []
        
        for example in dataset:
            prompt, label = self._create_prompt(example)
            if prompt is not None:
                prompts.append(prompt)
                labels.append(label)
                subject_names.append(example.get('subject_name', ''))
        
        return prompts, labels, subject_names

    @staticmethod
    def _create_prompt(example: Dict) -> Tuple[Optional[str], Optional[int]]:
        """Create a formatted prompt from a single example."""
        try:
            introduction = ("You are a medical expert and this is a multiple choice exam question. "
                          "Please respond with the integer index of the CORRECT answer only; [0,1,2,3].")
            
            question = example['question']
            
            # Parse options
            if isinstance(example['options'], str):
                options_dict = ast.literal_eval(example['options'])
            else:
                options_dict = example['options']
            
            options_filtered = {k: v for k, v in options_dict.items() if k != 'correct answer'}
            options_formatted = "Options: " + json.dumps(options_filtered)
            
            prompt = f"{introduction}\n\n{question}\n\n{options_formatted}"
            return prompt, example['correct_index']
            
        except Exception as e:
            logger.error(f"Error creating prompt: {e}")
            return None, None

    def filter_by_subject(self, prompts: List[str], labels: List[int], 
                        subject_names: List[str], subject_name: Optional[str] = None) -> Tuple[List[str], List[int]]:
        """Filter data by subject name."""
        if not subject_name:
            return prompts, labels
        
        subject_name = subject_name.lower()
        filtered_indices = [i for i, name in enumerate(subject_names) 
                          if name and name.lower() == subject_name]
        
        return ([prompts[i] for i in filtered_indices],
                [labels[i] for i in filtered_indices])