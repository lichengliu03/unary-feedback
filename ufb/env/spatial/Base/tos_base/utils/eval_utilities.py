"""
This file contains the functions for parsing the predicted string and evaluating the answer.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import re
import copy
import numpy as np


def extract_elements(
    pred: str,
    expected_type: type = str,
    clean_pattern: Optional[str] = None,
) -> Optional[Union[List[Any], Tuple[Any, Any]]]:
    """
    Extract elements from a raw input string, supporting both pairs and lists.
    
    Handles multiple input formats:
    - Parentheses format: (a, b) or (a, b, c, d)
    - Brackets format: [a, b] or [a, b, c, d]
    - Comma-separated list: a, b, c, d
    
    Args:
        pred (str): The raw input string
        expected_type (type): The expected type of elements (str, int, etc.)
        clean_pattern (Optional[str]): Optional regex pattern to remove from values before conversion
            
    Returns:
        Optional[Union[List[Any], Tuple[Any, Any]]]:
        - For pairs (force_pair=True): Tuple of (item1, item2) or None if extraction fails
        - For lists: List of extracted elements or None if extraction fails
    """
    if not pred or not isinstance(pred, str):
        return None
    
    try:
        content = pred
        
        # Check for parentheses format
        if '(' in pred and ')' in pred:
            match = re.search(r'\((.*?)\)', pred)
            if match:
                content = match.group(1)
        
        # Check for brackets format
        elif '[' in pred and ']' in pred:
            match = re.search(r'\[(.*?)\]', pred)
            if match:
                content = match.group(1)
        
        # Parse the elements
        items = []
        for item in content.split(','):
            # Strip whitespace and quotes
            cleaned = item.strip().strip("'\"")
            
            if not cleaned:  # Skip empty items
                continue
            
            # Apply additional cleaning if pattern provided
            if clean_pattern:
                cleaned = re.sub(clean_pattern, '', cleaned, flags=re.IGNORECASE)
            
            # Convert to the appropriate type
            if expected_type is int:
                try:
                    items.append(int(cleaned))
                except ValueError:
                    return None
            elif expected_type is float:
                try:
                    items.append(float(cleaned))
                except ValueError:
                    return None
            else:  # Default to string or other type
                items.append(expected_type(cleaned) if expected_type != str else cleaned)
        
        if not items:
            return None
        return items
        
    except Exception:
        # Handle any parsing errors gracefully
        return None




def exp_evaluate_fn(
        pred: str,
        relationships_to_query: List[Dict],
) -> bool:
    """
    Evaluate if the predicted string matches the ground truth relationships to query
    
    Args:
        pred (str): the raw predicted string
            1. relationship query:
                - Parentheses: (object1, object2)
                - Brackets: [object1, object2]
            2. termination:
                - "terminate"
        relationships_to_query (List[Dict]): the ground truth relationships to query
            - key: object1, object2, direction
        
    Returns:
        bool, True if the predicted string matches the ground truth relationships to query, False otherwise
    """
    
    if not pred or not isinstance(pred, str):
            return False
        
    # 1. First check if the answer is about termination
    if not relationships_to_query: # empty relationship_to_query means no unknown pairs for current room
        return "terminate" in pred.lower()
    
    # 2. Then check if the answer is about a relationship
    # 2.1 Extract the predicted pair
    pred_pair = extract_elements(pred, expected_type=str)
    if not pred_pair or len(pred_pair) != 2:
        return False
    
    pred_obj1, pred_obj2 = pred_pair
    
    # 2.2 Check if the predicted objects match any of the relationships to query
    for rel in relationships_to_query:
        gt_obj1 = rel['object1']
        gt_obj2 = rel['object2']
        
        # Check if the objects match (in either order)
        if ((pred_obj1.lower() == gt_obj1.lower() and pred_obj2.lower() == gt_obj2.lower()) or
            (pred_obj1.lower() == gt_obj2.lower() and pred_obj2.lower() == gt_obj1.lower())):
            return True
    
    return False



def tuple_eval_fn(pred: str, ground_truth: tuple) -> bool:
    """
    Evaluate if the predicted tuple matches the ground truth tuple
    - ground_truth: tuple of elements 
    - predicted: raw string, (elem1, elem2, ...), [elem1, elem2, ...], ...
    """
    if not pred or not isinstance(pred, str):
        return False
    elements = extract_elements(pred, expected_type=str)
    
    if not elements or len(elements) != len(ground_truth):
        return False
        
    # Compare elements directly
    for pred_elem, gt_elem in zip(elements, ground_truth):
        if pred_elem.lower() != gt_elem.lower():
            return False
    return True

def list_dir_eval_fn(
        pred: str,
        gt_list: List[tuple],
) -> bool:
    """
    Evaluate if the predicted list of directions matches the ground truth list of directions
    Parameters:
        pred (str): the raw predicted string, should follow the format:
            1. (<horiz>, <vert>)
            2. (<horiz>, <vert>)
            ...
        gt_list (List[tuple]): the ground truth list of direction tuples
    Returns:
        score (int): the number of correct predictions
    """
    # Parse predicted list
    pred_dirs_with_indices = []
    try:
        # Split by lines and remove empty lines
        lines = [line.strip() for line in pred.split('\n') if line.strip()]
        
        for line in lines:
            # Extract the index and direction
            if '. ' in line:
                parts = line.split('. ', 1)
                try:
                    idx = int(parts[0]) - 1  # Convert to 0-based index
                    direction = parts[1]
                    pred_dirs_with_indices.append((idx, direction))
                except ValueError:
                    # If index is not a valid integer, skip this line
                    continue
            else:
                # If no index format is found, skip this line
                continue
    except:
        return 0

    correct_count = 0
    # Check each prediction against the corresponding ground truth by index
    for idx, pred_dir in pred_dirs_with_indices:
        # Make sure the index is valid
        if 0 <= idx < len(gt_list):
            gt_dir = gt_list[idx]
            if tuple_eval_fn(pred_dir, gt_dir):
                correct_count += 1
    
    return correct_count
    
def obj_seq_eval_fn(
        pred: str,
        gt_object_sequence: List[str],
) -> bool:
    """
    Evaluate if the predicted object sequence matches the ground truth.
    
    Handles multiple input formats:
    - Array-like format: [a, b, c, d]
    - Comma-separated list: a, b, c, d
    - JSON-formatted array: ["a", "b", "c", "d"]
    
    Returns:
        bool: True if the predicted sequence matches the ground truth, False otherwise.
    """
    if not pred or not isinstance(pred, str):
        return False
    
    pred_objects = extract_elements(pred, str)
    
    if not pred_objects:
        return False
        
    # Quick length check
    if len(pred_objects) != len(gt_object_sequence):
        return False
    
    # Compare items (case-insensitive)
    for pred_obj, gt_obj in zip(pred_objects, gt_object_sequence):
        if pred_obj.lower() != gt_obj.lower():
            return False
            
    return True

def deg_seq_eval_fn(
        pred: str,
        gt_degree_sequence: List[int],
) -> bool:
    """
    Evaluate if the predicted sequence of degrees matches the ground truth.
    
    Handles multiple input formats:
    - Array-like format: [90, -45, 180, 30]
    - Comma-separated list: 90, -45, 180, 30
    - JSON-formatted array: [90, -45, 180, 30]
    
    Returns:
        bool: True if the predicted degree sequence matches the ground truth, False otherwise.
    """
    if not pred or not isinstance(pred, str):
        return False
    
    pred_degrees = extract_elements(pred, int, clean_pattern=r'[°degrees\s]')
    
    if not pred_degrees:
        return False
        
    # Check if the sequences match in length and values
    if len(pred_degrees) != len(gt_degree_sequence):
        return False

    for pred_deg, gt_deg in zip(pred_degrees, gt_degree_sequence):
        if pred_deg != gt_deg:
            return False
    
    return True


def obj_presence_eval_fn(pred: Any, answer: List[str]) -> Tuple[bool, Dict[str, Any]]:
    """
    Evaluate object presence task specifically.

    Handles multiple input formats:
    - book, lamp, chair
    - ["book", "lamp", "chair"]
    
    Args:
        pred: The predicted answer (string or list of object names)
        answer: The ground truth answer (list of object names)
        
    Returns:
        Tuple[bool, Dict[str, Any]]: (is_correct, info_dict with metrics)
    """
    # Parse predicted objects from text
    if not pred or not isinstance(pred, str):
        return False
    
    pred_objects = extract_elements(pred, str)
    
    if not pred_objects:
        return False
    
    # Ground truth objects (already lowercase)
    gt_objects = set(obj.lower() for obj in answer)
    pred_objects_set = set(obj.lower() for obj in pred_objects)
    
    # Calculate metrics
    correct_count = len(gt_objects.intersection(pred_objects_set))
    total_gt = len(gt_objects)
    precision = correct_count / len(pred_objects_set) if pred_objects_set else 0.0
    recall = correct_count / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    info = {
        "precision": precision,
        "recall": recall, 
        "f1": f1,
        "correct_count": correct_count,
        "total_gt": total_gt,
        "predicted_objects": list(pred_objects_set),
        "ground_truth_objects": list(gt_objects)
    }
    
    return correct_count == total_gt, info


def multi_choice_eval_fn(pred: str, answer: Union[List[str], str]) -> bool:
    """
    Evaluate if the predicted answer matches any of the valid answers (case-insensitive).
    
    Args:
        pred: The predicted answer (string)
        answer: List of valid answers
        
    Returns:
        bool: True if prediction matches any valid answer, False otherwise
    """
    if not pred or not isinstance(pred, str):
        return False
    answer = [answer] if isinstance(answer, str) else answer
    pred_cleaned = pred.strip().lower()
    return pred_cleaned in [ans.strip().lower() for ans in answer]


def e2a_eval_fn(pred: Any, answer: Any) -> Tuple[bool, Dict[str, Any]]:
    """
    Evaluate E2A (ego to allocentric) task specifically.
    
    Args:
        pred: The predicted answer (typically a string with coordinates)
        answer: The ground truth answer (list of coordinate tuples)
        
    Returns:
        Tuple[bool, Dict[str, Any]]: (is_correct, info_dict)
    """
    try:
        coord_pattern = r'[\(\[]?\s*(-?\d+)\s*,\s*(-?\d+)\s*[\)\]]?'
        matches = re.findall(coord_pattern, pred)
        if not matches or len(matches) != len(answer):
            return False, {"error": "No coordinates found in the response"}
        
        extracted_coords = [(int(x), int(y)) for x, y in matches]
        
        # Import here to avoid circular imports
        from ..core.graph import DirectionalGraph
        
        gt = copy.deepcopy(answer)
        gt.insert(0, (0, 0))
        gt_v_matrix, gt_h_matrix = DirectionalGraph.create_graph_from_coordinates(gt)

        pred_coords = copy.deepcopy(extracted_coords)
        pred_coords.insert(0, (0, 0))
        pred_v_matrix, pred_h_matrix = DirectionalGraph.create_graph_from_coordinates(pred_coords)

        if np.allclose(pred_v_matrix, gt_v_matrix) and np.allclose(pred_h_matrix, gt_h_matrix):
            return True, {}
        else:
            return False, {}
    except Exception as e:
        print("Error in E2AEvaluationTask", e)
        return False, {}


if __name__ == "__main__":
    # Test cases for list_dir_eval_fn
    
    # # Test case 1: Basic correct case
    # test_answer_1 = "1. (left, above)\n2. (right, below)\n3. (right, same)"
    # test_gt_1 = ["(left, above)", "(right, below)", "(right, same)"]
    # result_1 = list_dir_eval_fn(test_answer_1, test_gt_1)
    # print(f"Test 1 result: {result_1}")  # Should be True
    
    # # Test case 2: Incorrect order
    # test_answer_2 = "1. (right, below)\n2. (left, above)\n3. (right, same)"
    # test_gt_2 = ["(left, above)", "(right, below)", "(right, same)"]
    # result_2 = list_dir_eval_fn(test_answer_2, test_gt_2)
    # print(f"Test 2 result: {result_2}")  # Should be False
    
    # # Test case 3: Different formatting but correct content
    # test_answer_3 = "1. (Left, Above)\n2. (Right, Below)\n3. (Right, Same)"
    # test_gt_3 = ["(left, above)", "(right, below)", "(right, same)"]
    # result_3 = list_dir_eval_fn(test_answer_3, test_gt_3)
    # print(f"Test 3 result: {result_3}")  # Should be True

    # # Test case 4: Incorrect number of predictions
    # test_answer_4 = "1. (left, above)\n4. (left, below)\n3. (right, same)\n2. (right, below)"
    # test_gt_4 = ["(left, above)", "(right, below)", "(right, same)"]
    # result_4 = list_dir_eval_fn(test_answer_4, test_gt_4)
    # print(f"Test 4 result: {result_4}")  # Should be False
    
    pass