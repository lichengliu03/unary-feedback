import re
from typing import Tuple
import hashlib

# Reusable labels for formatting and parsing
THINK_LABEL = "THINK:"
ANSWER_LABEL = "FINAL ANSWER:"

def parse_llm_response(text: str, enable_think: bool = True) -> Tuple[str, str, bool]:
    """Parse LLM response for optional THINK and required FINAL ANSWER blocks.

    Expected format (labels on their own line). Parser tolerates content on the same line too:
    THINK:\n
    <think_content>    or    THINK: <think_content>
    FINAL ANSWER:\n
    <answer_content>  or    FINAL ANSWER: <answer_content>

    Returns: (think_content, answer_content, parsed_ok). parsed_ok is True if an answer was extracted.
    """
    if not isinstance(text, str):
        return "", "", False

    # Prefer new header-style format (labels can be mid-line or on their own line)
    think_re = re.compile(rf"(?is){re.escape(THINK_LABEL)}\s*(.*?)(?={re.escape(ANSWER_LABEL)}|\Z)")
    answer_re = re.compile(rf"(?is){re.escape(ANSWER_LABEL)}\s*(.*)\Z")

    think_match = think_re.search(text)
    answer_match = answer_re.search(text)
    think_content = (think_match.group(1).strip() if think_match else "")
    answer_content = (answer_match.group(1).strip() if answer_match else "")

    # Backward-compat: fall back to legacy <think>/<answer> tags if headers not found
    if not answer_content:
        legacy_ans = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        if legacy_ans:
            answer_content = legacy_ans.group(1).strip()
    if not think_content:
        legacy_think = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
        if legacy_think:
            think_content = legacy_think.group(1).strip()

    # If answer still missing, treat whole text as the answer (answer is critical)
    if not answer_content:
        answer_content = text.strip()

    # Clean GLM-4.5V box tokens if present
    if answer_content:
        answer_content = (
            answer_content.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "").strip()
        )

    if not enable_think:
        return "", answer_content, bool(answer_content)

    parsed_ok = bool(answer_content)
    return think_content, answer_content, parsed_ok

def hash(input_str: str) -> str:
    """Generate a stable hash for the given input string."""
    return hashlib.sha256(input_str.encode('utf-8')).hexdigest()[:16]

def format_llm_output(think_content: str, answer_content: str, enable_think: bool = True) -> str:
    """Format output with headers based on enable_think.

    THINK:\n<think_content>\nFINAL ANSWER:\n<answer_content>
    """
    if enable_think:
        return f"{THINK_LABEL}\n{think_content}\n{ANSWER_LABEL}\n{answer_content}"
    return f"{ANSWER_LABEL}\n{answer_content}"

if __name__ == "__main__":
    # Simple parser sanity checks
    llm_response = f"{THINK_LABEL}\nI will move then observe.{ANSWER_LABEL}\nActions: [JumpTo(table), Observe()]"
    t, a, ok = parse_llm_response(llm_response, enable_think=True)
    print('think:', t)
    print('answer:', a)
    print('ok:', ok)