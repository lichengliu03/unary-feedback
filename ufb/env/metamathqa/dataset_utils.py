from functools import lru_cache

from datasets import load_dataset


@lru_cache(maxsize=None)
def load_filtered_metamathqa_dataset(dataset_path: str, cache_dir: str, split: str):
    """Load and filter the MetaMathQA split once per process."""
    dataset = load_dataset(path=dataset_path, cache_dir=cache_dir)
    return dataset[split].filter(lambda example: example["type"].startswith("MATH_"))
