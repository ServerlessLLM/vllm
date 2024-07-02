CUDA_VISIBLE_DEVICES=0,1 python save_sllm_state.py \
    --model /mnt/raid0sata1/huggingface/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6 \
    --tensor-parallel-size 4 \
    --output /mnt/raid0nvme1/xly/test_data/vllm/opt-125m

CUDA_VISIBLE_DEVICES=0,1 python load_sllm_state.py \
    --model /home/fuji/.cache/huggingface/hub/models--facebook--opt-1.3b/snapshots/3f5c25d0bc631cb57ac65913f76e22c2dfb61d62 \
    --tensor-parallel-size 2 \
    --output /home/fuji/sllm_models/opt-1.3b