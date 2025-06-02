import tensorflow as tf
import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import ndcg_score
from typing import List, Tuple, Dict, Set
import warnings

def extract_common_elements(list1: List[Tuple], list2: List[Tuple]) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Extract common elements from two ranked lists based on IDs.
    Works with both tensors and normal numbers (int/float) in the tuples.
    
    Args:
        list1, list2: Lists of tuples (id, score), where id can be a tensor or a number.
    
    Returns:
        Tuple of filtered lists containing only common elements, preserving original order.
    """
    def get_id(item):
        # Handle case where item[0] is a tensor
        if hasattr(item[0], 'numpy'):
            return float(item[0].numpy())
        # Handle case where item[0] is a number (int/float)
        else:
            return float(item[0])
    
    # Extract IDs from both lists
    ids1 = {get_id(item) for item in list1}
    ids2 = {get_id(item) for item in list2}
    
    # Find common IDs
    common_ids = ids1.intersection(ids2)
    
    # Filter lists to keep only common elements
    filtered_list1 = [item for item in list1 if get_id(item) in common_ids]
    filtered_list2 = [item for item in list2 if get_id(item) in common_ids]
    
    return filtered_list1, filtered_list2

def create_ranking_mappings(filtered_list1: List[Tuple], filtered_list2: List[Tuple]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create ranking mappings for correlation analysis.
    Works with both tensors and normal numbers (int/float) in the tuples.
    
    Returns:
        ranks1: Ranking positions of items in list1 (1-based)
        ranks2: Ranking positions of items in list2 (1-based) 
        scores2: Actual scores from list2 (for NDCG calculation)
    """
    def get_id(item):
        # Handle case where item[0] is a tensor
        if hasattr(item[0], 'numpy'):
            return float(item[0].numpy())
        # Handle case where item[0] is a number (int/float)
        else:
            return float(item[0])
    
    def get_score(item):
        # Handle case where item[1] is a tensor
        if hasattr(item[1], 'numpy'):
            return float(item[1].numpy())
        # Handle case where item[1] is a number (int/float)
        else:
            return float(item[1])
    
    # Create ID to rank mapping for both lists
    id_to_rank1 = {get_id(item): idx + 1 for idx, item in enumerate(filtered_list1)}
    id_to_rank2 = {get_id(item): idx + 1 for idx, item in enumerate(filtered_list2)}
    id_to_score2 = {get_id(item): get_score(item) for item in filtered_list2}
    
    # Get common IDs in the order they appear in list1
    common_ids_ordered = [get_id(item) for item in filtered_list1]
    
    # Create aligned ranking arrays
    ranks1 = np.array([id_to_rank1[id_] for id_ in common_ids_ordered])
    ranks2 = np.array([id_to_rank2[id_] for id_ in common_ids_ordered])
    scores2 = np.array([id_to_score2[id_] for id_ in common_ids_ordered])
    
    return ranks1, ranks2, scores2

def calculate_correlation_metrics(list1: List[Tuple], list2: List[Tuple]) -> Dict[str, float]:
    """
    Calculate Kendall Tau, Spearman ρ, and NDCG between two ranked lists.
    
    Args:
        list1, list2: Lists of tuples (id_tensor, score_tensor) ranked by score (descending)
    
    Returns:
        Dictionary with correlation metrics
    """
    # Extract common elements
    filtered_list1, filtered_list2 = extract_common_elements(list1, list2)
    
    if len(filtered_list1) <= 1:
        return {
            'common_elements': len(filtered_list1),
            'kendall_tau': np.nan,
            'kendall_p_value': np.nan,
            'spearman_rho': np.nan,
            'spearman_p_value': np.nan,
            'ndcg': np.nan
        }
    
    # Create ranking mappings
    ranks1, ranks2, scores2 = create_ranking_mappings(filtered_list1, filtered_list2)
    
    # Calculate Kendall Tau
    kendall_tau, kendall_p = kendalltau(ranks1, ranks2)
    
    # Calculate Spearman correlation
    spearman_rho, spearman_p = spearmanr(ranks1, ranks2)
    
    # Calculate NDCG
    # For NDCG, we need to convert rankings to relevance scores
    # We'll use the actual scores from list2 as true relevance
    # and create predicted relevance based on list1's ranking
    
    # Convert rankings to relevance scores (higher rank = lower relevance score)
    # We'll use a simple transformation: max_rank + 1 - rank
    max_rank = len(ranks1)
    predicted_relevance = max_rank + 1 - ranks1  # Higher for better ranks in list1
    true_relevance = scores2  # Actual scores from list2
    
    # NDCG expects relevance scores in descending order of prediction
    # We need to reshape for sklearn's ndcg_score function
    ndcg = ndcg_score([true_relevance], [predicted_relevance])
    
    return {
        'common_elements': len(filtered_list1),
        'kendall_tau': kendall_tau,
        'kendall_p_value': kendall_p,
        'spearman_rho': spearman_rho,
        'spearman_p_value': spearman_p,
        'ndcg': ndcg
    }

def print_comparison_results(results: Dict[str, float]):
    """Print formatted comparison results."""
    print("Ranked Lists Comparison Results:")
    print("=" * 40)
    print(f"Common elements: {results['common_elements']}")
    print()
    print("Correlation Metrics:")
    print(f"Kendall Tau: {results['kendall_tau']:.4f} (p-value: {results['kendall_p_value']:.4f})")
    print(f"Spearman ρ:  {results['spearman_rho']:.4f} (p-value: {results['spearman_p_value']:.4f})")
    print(f"NDCG:        {results['ndcg']:.4f}")
    print()

def print_comparison_interpretation():
    """Print formatted information on how to interpret the results"""
    print("Interpretation:")
    print("- Values closer to 1 indicate strong positive correlation")
    print("- Values closer to -1 indicate strong negative correlation") 
    print("- Values closer to 0 indicate weak correlation")
    print("- NDCG ranges from 0 to 1, with 1 being perfect ranking")