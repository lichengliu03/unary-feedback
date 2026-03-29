from huggingface_hub import snapshot_download

snapshot_download(  
	repo_id="ZihanWang314/exp1_MetamathQA_global_step_200",  
	repo_type="model",  
	local_dir="/workspace/loaded_checkpoints/exp1/metamathqa",  
)