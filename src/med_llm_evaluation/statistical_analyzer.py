from typing import Dict, List, Tuple
from collections import Counter
import numpy as np
import scipy.stats

class StatisticalAnalyzer:
    """Handles statistical analysis of model performance."""
    
    @staticmethod
    def analyze(y_true: List[int], 
                y_pred: List[int], 
                alpha: float = 0.05) -> Dict:
        """
        Analyze if model performance is significantly better than random.
        
        Args:
            y_true (List[int]): True labels
            y_pred (List[int]): Predicted labels
            alpha (float): Significance level
            
        Returns:
            Dict: Statistical analysis results
        """
        # Basic metrics
        n_samples = len(y_true)
        n_correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        observed_accuracy = n_correct / n_samples
        
        # Calculate baselines
        random_prob, class_proportions = StatisticalAnalyzer._calculate_random_baseline(y_true)
        
        # Statistical test
        p_value = scipy.stats.binomtest(n_correct, n_samples, p=random_prob, 
                                      alternative='greater').pvalue
        
        # Effect size
        h = 2 * (np.arcsin(np.sqrt(observed_accuracy)) - 
                np.arcsin(np.sqrt(random_prob)))
        
        # Interpret results
        is_significant = p_value < alpha
        effect_size = StatisticalAnalyzer._interpret_effect_size(h)
        
        # Minimum needed for significance
        min_successes = StatisticalAnalyzer._find_min_successes(n_samples, random_prob, alpha)
        min_accuracy = min_successes / n_samples
        
        results = {
            'better_than_random': is_significant,
            'p_value': p_value,
            'observed_accuracy': observed_accuracy,
            'effect_size': h,
            'effect_size_interpretation': effect_size,
            'n_samples': n_samples,
            'n_correct': n_correct,
            'min_correct': min_successes,
            'min_accuracy_needed': min_accuracy,
            'random_baseline': random_prob,
            'class_distribution': class_proportions
        }
        
        # Add human-readable summary
        results['summary'] = StatisticalAnalyzer._create_summary(results, alpha)
        
        return results
    
    @staticmethod
    def _calculate_random_baseline(y_true: List[int]) -> Tuple[float, Dict[int, float]]:
        """Calculate random baseline accuracy based on class distribution."""
        class_counts = Counter(y_true)
        total = len(y_true)
        
        proportions = {k: v/total for k, v in class_counts.items()}
        random_baseline = sum(p*p for p in proportions.values())
        
        return random_baseline, proportions
    
    @staticmethod
    def _find_min_successes(n: int, p: float, alpha: float) -> int:
        """Find minimum successes needed for significance."""
        left, right = int(n * p), n
        
        while left <= right:
            mid = (left + right) // 2
            p_value = scipy.stats.binomtest(mid, n, p, alternative='greater').pvalue
            
            if p_value <= alpha:
                if mid == left or scipy.stats.binomtest(mid - 1, n, p, 
                                                alternative='greater').pvalue > alpha:
                    return mid
                right = mid - 1
            else:
                left = mid + 1
        
        return left
    
    @staticmethod
    def _interpret_effect_size(h: float) -> str:
        """Interpret Cohen's h effect size."""
        if abs(h) < 0.2:
            return 'negligible'
        elif abs(h) < 0.5:
            return 'small'
        elif abs(h) < 0.8:
            return 'medium'
        else:
            return 'large'
    
    @staticmethod
    def _create_summary(results: Dict, alpha: float) -> str:
        """Create human-readable summary of results."""
        dist_desc = "\nClass Distribution:\n"
        for class_label, prop in sorted(results['class_distribution'].items()):
            count = int(prop * results['n_samples'])
            dist_desc += (f"Class {class_label}: {prop:.3f} "
                        f"({count}/{results['n_samples']} samples)\n")
        
        return f"""
Performance Assessment:
----------------------
Observed Accuracy: {results['observed_accuracy']:.3f} ({results['n_correct']}/{results['n_samples']})
Random Baseline: {results['random_baseline']:.3f} (based on class distribution)
P-value: {results['p_value']:.4f}
Effect Size (Cohen's h): {results['effect_size']:.3f} ({results['effect_size_interpretation']})

{dist_desc}
Statistical Significance:
The model {'is' if results['better_than_random'] else 'is not'} performing significantly better than the random baseline
(p{' < ' if results['p_value'] < alpha else ' = '}{results['p_value']:.4f})

For {results['n_samples']} samples, needed {results['min_accuracy_needed']:.3f} accuracy ({results['min_correct']} correct)
for statistical significance at α={alpha}
"""