# Hugging Face Upload

1. Login once:

```bash
hf auth login
```

2. Run:

```bash
python scripts/utils/convert_and_upload_checkpoints_to_hf.py \
  --checkpoint-parent-dir outputs/checkpoints/exp1_MetamathQA \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --remove-temp-root-if-empty
```

This will find each `global_step_*/actor` checkpoint, convert it to a temporary HF model, upload it as a public repo, and delete the temporary HF directory after a successful upload.
