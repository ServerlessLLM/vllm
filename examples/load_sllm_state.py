"""
Saves each worker's model state dict directly to a checkpoint, which enables a
fast load path for large tensor-parallel models where each worker only needs to
read its own shard rather than the entire checkpoint.

Example usage:

python save_sharded_state.py \
    --model /path/to/load \
    --quantization deepspeedfp \
    --tensor-parallel-size 8 \
    --output /path/to/save

Then, the model can be loaded with

llm = LLM(
    model="/path/to/save",
    load_format="sharded_state",
    quantization="deepspeedfp",
    tensor_parallel_size=8,
)
"""
import argparse
import dataclasses
import os
import shutil
from pathlib import Path

from vllm import LLM, EngineArgs

parser = argparse.ArgumentParser()
EngineArgs.add_cli_args(parser)
parser.add_argument("--output",
                    "-o",
                    required=True,
                    type=str,
                    help="path to output checkpoint")

if __name__ == "__main__":
    args = parser.parse_args()
    # main(args)
    
    llm = LLM(
        model=args.output,
        load_format="serverless_llm",
        # load_format="sharded_state",
        gpu_memory_utilization=0.9,
        distributed_executor_backend="mp",
        max_model_len = 512,
        tensor_parallel_size=args.tensor_parallel_size,
        # num_gpu_blocks_override=128,
    )
    
    input_text = "Explain thread and process in python."
    
    print(llm.generate(input_text))
