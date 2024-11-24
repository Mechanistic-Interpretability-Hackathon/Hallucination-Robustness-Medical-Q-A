"""Medical LLM Evaluation package for assessing LLM performance on medical questions."""

from src.med_llm_evaluation.medical_evaluator import MedicalLLMEvaluator
from src.med_llm_evaluation.data_handler import DataHandler
from src.med_llm_evaluation.llm_evaluator import LLMEvaluator
from src.med_llm_evaluation.statistical_analyzer import StatisticalAnalyzer

__all__ = [
    'MedicalLLMEvaluator',
    'DataHandler',
    'LLMEvaluator',
    'StatisticalAnalyzer'
]

__version__ = '0.1.0'