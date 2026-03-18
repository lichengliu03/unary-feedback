#!/usr/bin/env python3
"""Download all required datasets for UFB training."""

import os
from huggingface_hub import snapshot_download


def main():
    os.makedirs("data", exist_ok=True)

    # 1. Planning envs (sokoban, frozen_lake, bandit)
    print("Downloading planning env data (sokoban, frozen_lake, bandit)...")
    snapshot_download(
        repo_id="ZihanWang314/ragen-datasets",
        repo_type="dataset",
        local_dir="data",
        local_dir_use_symlinks=False,
    )
    print("Done.\n")

    # 2. MetaMathQA
    print("Downloading MetaMathQA...")
    import datasets
    ds = datasets.load_dataset("meta-math/MetaMathQA")
    save_dir = "data/meta-math___meta_math_qa"
    os.makedirs(save_dir, exist_ok=True)
    ds.save_to_disk(save_dir)
    print("Done.\n")

    print("All datasets downloaded to data/")


if __name__ == "__main__":
    main()
