from typing import Dict, Any, List
import numpy as np
from scipy.stats import pearsonr


def compute_correlation_metrics(env_data_list: Dict, exp_type: str = 'active') -> Dict[str, Any]:
    """
    Compute correlations between cognitive map metrics and evaluation metrics, information gain metrics.

    Args:
        env_data_list: Environment data list
        exp_type: Task type, 'active' or 'passive'

    Returns:
        Dictionary containing all correlation analysis results
    """
    assert isinstance(env_data_list, list) and len(env_data_list) > 0, "env_data_list must be a non-empty list"
    if exp_type == 'passive':
        return {}

    # First pass: collect all existing task names
    all_task_names = set()
    for s in env_data_list:
        metrics = s.get('metrics') or {}
        evaluation_metric = metrics.get('evaluation') or {}
        if isinstance(evaluation_metric, dict):
            per_task = evaluation_metric.get('per_task', {})
            all_task_names.update(per_task.keys())

    last_global_vs_gt_fulls = []
    evaluation_metric_list = {'avg_accuracy': []}
    # Initialize all task names with empty lists
    for task_name in all_task_names:
        evaluation_metric_list[task_name] = []
    last_infogains = []

    for s in env_data_list:
        metrics = s.get('metrics') or {}
        cogmap_metric = metrics.get('cogmap') or {}
        evaluation_metric = metrics.get('evaluation') or {}
        exploration_metric = metrics.get('exploration', {}) or {}
        if not isinstance(cogmap_metric, dict) or not isinstance(evaluation_metric, dict):
            continue

        # Extract last_global_vs_gt_full metric
        exploration = cogmap_metric.get('exploration', {})
        correctness = exploration.get('correctness', {})
        last_global_full = correctness.get('last_global_vs_gt_full', {}) or {}
        # Only keep samples with valid last_global_vs_gt_full
        if not last_global_full or 'overall' not in last_global_full:
            continue
        overall_cogmap = last_global_full.get('overall')

        # Extract last_infogain metric
        last_infogain = exploration_metric.get('final_information_gain')

        if isinstance(overall_cogmap, (int, float)) and not np.isnan(overall_cogmap):
            last_global_vs_gt_fulls.append(float(overall_cogmap))

            # Extract evaluation metrics
            # Overall accuracy
            avg_accuracy = (evaluation_metric.get('overall') or {}).get('avg_accuracy')
            evaluation_metric_list['avg_accuracy'].append(float(avg_accuracy) if isinstance(avg_accuracy, (int, float)) else None)

            # Accuracy for each task - fill missing tasks with None
            per_task = evaluation_metric.get('per_task') or {}
            for task_name in all_task_names:
                task_acc = (per_task.get(task_name) or {}).get('avg_accuracy')
                evaluation_metric_list[task_name].append(float(task_acc) if isinstance(task_acc, (int, float)) else None)

            # Add information gain data (aligned)
            last_infogains.append(float(last_infogain) if isinstance(last_infogain, (int, float)) and not np.isnan(last_infogain) else None)

    cogmap_acc_correlations = {}
    for task_name, evaluation_values in evaluation_metric_list.items():
        cogmap_acc_correlations[task_name] = calculate_pearson_correlation(last_global_vs_gt_fulls, evaluation_values)

    cogmap_infogain_correlation = calculate_pearson_correlation(last_global_vs_gt_fulls, last_infogains)

    return {
        'cogmap_acc_correlations': cogmap_acc_correlations,
        'cogmap_infogain_correlation': cogmap_infogain_correlation,
        'last_global_vs_gt_fulls': last_global_vs_gt_fulls,
        'last_infogains': last_infogains,
        'avg_acc_metrics': evaluation_metric_list.get('avg_accuracy', []),
        'n_samples': len(last_global_vs_gt_fulls)
    }


def calculate_pearson_correlation(x: List[float], y: List[float]) -> Dict[str, Any]:
    try:
        # Pairwise filter (ignore None/NaN), allow length mismatch by zipping
        valid_pairs = []
        for xi, yi in zip(x or [], y or []):
            if xi is None or yi is None:
                continue
            if isinstance(xi, (int, float)) and isinstance(yi, (int, float)) and not np.isnan(xi) and not np.isnan(yi):
                valid_pairs.append((float(xi), float(yi)))
        if len(valid_pairs) < 2:
            return {
                'pearson_r': None,
                'p_value': None,
                'significant': False,
                'n_samples': len(valid_pairs)
            }
        x_valid, y_valid = zip(*valid_pairs)
        corr_coef, p_value = pearsonr(x_valid, y_valid)
        return {
            'pearson_r': float(corr_coef),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),
            'n_samples': len(valid_pairs)
        }
    except Exception as e:
        return {
            'pearson_r': None,
            'p_value': None,
            'significant': False,
            'n_samples': 0,
            'error': str(e)
        }

