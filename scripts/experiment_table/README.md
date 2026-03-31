## Experiment Table Scripts

This directory contains runnable `.sh` launchers generated from the experiment table.

- `HotpotQA` and `MetaMathQA` wrappers call `_train_standard_experiment.sh`
- `WebShop` reuses the existing benchmark-specific launcher in [`scripts/webshop/train_qwen25_3b.sh`](/u/ylin30/unary-feedback/scripts/webshop/train_qwen25_3b.sh)
- The duplicated `MetaMathQA + Qwen2.5-3B + specific` row is emitted as `rep1` and `rep2`

Current files:

- `hotpotqa_qwen25_3b_one_bit.sh`
- `webshop_qwen25_3b_one_bit.sh`
- `metamathqa_qwen25_3b_specific_rep1.sh`
- `metamathqa_qwen25_3b_specific_rep2.sh`
- `metamathqa_qwen25_1p5b_one_bit.sh`
- `metamathqa_qwen25_7b_one_bit.sh`
- `metamathqa_llama32_3b_one_bit.sh`
- `metamathqa_llama32_3b_single_turn_grpo.sh`
- `metamathqa_gemma3_4b_one_bit.sh`
- `metamathqa_phi4_mini_one_bit.sh`
- `metamathqa_llama32_3b_no_feedback.sh`
- `metamathqa_gemma3_4b_no_feedback.sh`
- `metamathqa_phi4_mini_no_feedback.sh`
