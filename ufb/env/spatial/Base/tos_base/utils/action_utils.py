"""
Utility functions for converting action results to text observations.
"""
from typing import List
from ..actions import ActionResult


def action_results_to_text(action_results: List[ActionResult], placeholder: str = None) -> str:
    """Convert list of ActionResults to text observation.
    
    Args:
        action_results: List of ActionResult objects from action execution
    
    Returns:
        Text observation string
    """
    assert action_results, "action_results is empty"
    messages = []
    for result in action_results:
        # Check if this is an observe action result
        if placeholder and 'observe' in result.action_type:
            # Replace observe result with image placeholder
            messages.append(f"You observe: {placeholder}.")
        else:
            # Keep original message for non-observe actions
            messages.append(result.message)

    return " ".join(messages)