from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="zihanwang314/unary-feedback-checkpoints",
    repo_type="model",
    token="",
    local_dir="/workspace/loaded_checkpoints/metamathqa_qwen25_3b_specific_5attempts_5turns",
    allow_patterns=[
        "ufb_exp/metamathqa_qwen25_3b_specific_5attempts_5turns/global_step_200/*",
    ]
)

