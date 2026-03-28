from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="zihanwang314/unary-feedback-checkpoints",
    repo_type="model",
    token="",
    local_dir="./projects/bflz/loaded_checkpoints/hotpotqa_qwen25_3b_one_bit_5attempts_1turn",
    allow_patterns=[
        "ufb_exp/hotpotqa_qwen25_3b_one_bit_5attempts_1turn/**"
    ]
)
