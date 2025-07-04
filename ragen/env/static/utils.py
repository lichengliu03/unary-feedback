import re
import string
from typing import Dict, Any, Optional, List, Tuple, Callable


############################Tool Fuctions############################
def normalize_text(text: str) -> str:
    """Normalize text by removing whitespace, punctuation, and converting to lowercase."""
    text = text.lower()
    text = re.sub(r'\s+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def extract_answer_from_text(text: str) -> str:
    """Extract answer from text with various patterns."""
    patterns = [
        r"The answer is:?\s*(.*?)(?:\n|$)",
        r"Answer:?\s*(.*?)(?:\n|$)",
        r"Final answer:?\s*(.*?)(?:\n|$)",
        r"Therefore,\s*(.*?)(?:\n|$)",
        r"Thus,\s*(.*?)(?:\n|$)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # If no pattern matches, return the last line as a fallback
    lines = text.strip().split('\n')
    return lines[-1].strip()
# ====== Dataset Processors ======

def process_metamathqa(item: Dict[str, Any]) -> Tuple[str, str]:
    """Process MetaMathQA dataset item."""
    question = item["query"]
    answer = extract_answer_from_text(item["response"])
    return question, answer

def process_gsm8k(item: Dict[str, Any]) -> Tuple[str, str]:
    """Process GSM8K dataset item."""
    question = item["question"]
    answer = item["answer"]
    answer=answer.split("####")[1].strip().lower()
    return question, answer

def process_theoremqa(item: Dict[str, Any]) -> Tuple[str, str]:
    """Process TheoremQA dataset item with image support."""
    question = item["Question"]
    answer = str(item["Answer"])
    
    # Handle images if present
    if item.get("Picture") is not None:
        # For environments with multimodal support, add image token
        # The image will be handled separately in the multimodal pipeline
        question = f"<image>\n{question}"
    
    return question, answer

def process_mmlu(item: Dict[str, Any]) -> Tuple[str, str]:
    """Process MMLU dataset with multiple choice format."""
    question = item['question']
    choices = [item['choices'][i] for i in range(len(item['choices']))]
    formatted_question = question + "\n" + "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
    answer = chr(65 + item['answer'])  # Convert to A, B, C, D format
    return formatted_question, answer

def process_gpqa(item: Dict[str, Any]) -> Tuple[str, str]:
    """Process GPQA dataset item."""
    question = item["Question"]
    answer = extract_answer_from_text(item["Correct Answer"])
    return question, answer

def process_hotpotqa(item: Dict[str, Any]) -> Tuple[str, str]:
    """Process HotpotQA dataset item."""
    question = item["question"]
    answer = item["answer"]
    
    # Optionally add context information to the question
    if "context" in item and item["context"]:
        context_info = []
        titles = item["context"]["title"]
        sentences = item["context"]["sentences"]
        
        # Combine context from multiple documents
        for i, (title, doc_sentences) in enumerate(zip(titles, sentences)):
            if len(context_info) < 3:  # Limit context to avoid too long prompts
                context_info.append(f"Document {i+1} ({title}): {' '.join(doc_sentences[:2])}")  # First 2 sentences
        
        if context_info:
            question = f"Context: {' | '.join(context_info)}\n\nQuestion: {question}"
    
    return question, answer

def process_musique(item: Dict[str, Any]) -> Tuple[str, str]:
    """Process MuSiQue dataset item."""
    question = item["question"]
    answer = item["answer"]
    return question, answer

def process_mmlu_pro(item: Dict[str, Any]) -> Tuple[str, str]:
    """Process MMLU-Pro dataset item (10 options, answer letter A-J)."""
    question = item["question"]
    # options column may be 'options' list or 'choices'.
    options = item.get("options", item.get("choices"))
    if options is None:
        # Fallback: build empty list
        options = []
    formatted_question = question
    for idx, opt in enumerate(options):
        formatted_question += f"\n{chr(65+idx)}. {opt}"
    # answer may be provided as letter or index
    if "answer" in item:
        answer_field = item["answer"]
        if isinstance(answer_field, str):
            answer = answer_field.strip().upper()[:1]
        else:
            # numeric index -> letter
            answer = chr(65 + int(answer_field))
    else:
        # fallback using answer_index
        answer_idx = int(item.get("answer_index", 0))
        answer = chr(65 + answer_idx)
    return formatted_question, answer

# ====== Scoring Functions ======

def compute_score_exact_match(prediction: str, label: str) -> Dict[str, Any]:
    """Basic exact match after normalization."""
    norm_pred = normalize_text(prediction)
    norm_label = normalize_text(label)
    
    is_correct = norm_pred == norm_label
    is_valid = len(norm_pred) > 0  # Simple validity check
    
    return {
        "is_correct": is_correct,
        "is_valid": is_valid,
        "normalized_prediction": norm_pred,
        "normalized_label": norm_label
    }

def compute_score_numeric(prediction: str, label: str) -> Dict[str, Any]:
    """Extract numeric values and compare them."""
    # Extract the first numeric value from both prediction and label
    pred_match = re.search(r'(\d+(?:\.\d+)?)', prediction)
    label_match = re.search(r'(\d+(?:\.\d+)?)', label)
    
    is_valid = pred_match is not None
    
    if pred_match and label_match:
        pred_answer = pred_match.group(0)
        label_answer = label_match.group(0)
        
        try:
            is_correct = float(pred_answer) == float(label_answer)
        except ValueError:
            is_correct = False
    else:
        is_correct = False
    
    # Also try text match as fallback
    text_match = normalize_text(prediction) == normalize_text(label)
    is_correct = is_correct or text_match
    
    return {
        "is_correct": is_correct,
        "is_valid": is_valid,
        "numeric_match": is_correct and not text_match,
        "text_match": text_match
    }

def compute_score_multiple_choice(prediction: str, label: str) -> Dict[str, Any]:
    """Score multiple choice answers (A, B, C, D)."""
    pred_match = re.search(r'([A-D])', prediction.upper())
    label_match = re.search(r'([A-D])', label.upper())
    
    is_valid = pred_match is not None
    
    if pred_match and label_match:
        pred_choice = pred_match.group(0)
        label_choice = label_match.group(0)
        is_correct = pred_choice == label_choice
    else:
        # Fallback to text comparison
        is_correct = normalize_text(prediction) == normalize_text(label)
    
    return {
        "is_correct": is_correct,
        "is_valid": is_valid,
        "extracted_prediction": pred_match.group(0) if pred_match else None,
        "extracted_label": label_match.group(0) if label_match else None
    }
    
##########################registration###########################
REGISTERD_STATIC_ENV = {
    "metamathqa": {
        "config": {
            "path": "meta-math/MetaMathQA",
        },
        "processor": process_metamathqa,
        "compute_score": compute_score_exact_match
    },
    "gsm8k": {
        "config": {
            "path": "openai/gsm8k",
            "name":"main"
        },
        "processor": process_gsm8k,
        "compute_score": compute_score_numeric
    },
    "theoremqa": {
        "config": {
            "path": "TIGER-Lab/TheoremQA",
        },
        "processor": process_theoremqa,
        "compute_score": compute_score_numeric
    },
    "mmlu": {
        "config": {
            "path": "cais/mmlu",
            "name": "abstract_algebra",
        },
        "processor": process_mmlu,
        "compute_score": compute_score_multiple_choice
    },
    # "gpqa":{
    #     "config": {
    #         "path": "Idavidrein/gpqa",
    #         "name": "gpqa_main",
    #     },
    #     "processor": process_gpqa,
    #     "compute_score": compute_score_exact_match
    # }
    "hotpotqa": {
        "config": {
            "path": "hotpotqa/hotpot_qa",
            "name": "distractor",
        },
        "processor": process_hotpotqa,
        "compute_score": compute_score_exact_match
    },
    "musique": {
        "config": {
            "path": "dgslibisey/MuSiQue",
            "name": "default",
        },
        "processor": process_musique,
        "compute_score": compute_score_exact_match
    },
    "mmlu_pro": {
        "config": {
            "path": "TIGER-Lab/MMLU-Pro",
            "name": "default",
        },
        "processor": process_mmlu_pro,
        "compute_score": compute_score_multiple_choice
    }
}
