import os
import json
import sys
import argparse

sys.path.insert(0, "/disk/revision/vllm")  # vllm 소스 디렉토리의 상위 경로

from lm_eval import evaluator
from vllm.config import CompilationConfig
import logging

logging.basicConfig(
    level=logging.INFO,  # or logging.DEBUG for more detail
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Name of model to load")
    parser.add_argument("--gpu", type=str, required=True, help="GPU device ID to use (e.g., 0)")
    parser.add_argument("--task", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_FLASH_ATTN_VERSION"] = "3"
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"  # if needed
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    compilation_config = CompilationConfig(
        level=3,
        custom_ops=["none"],
        splitting_ops=["vllm.unified_attention","vllm.unified_attention_with_output"],
        use_inductor=True,
        full_cuda_graph=True,
        compile_sizes=[],
        use_cudagraph=True,
        cudagraph_num_of_warmups=1,
        cudagraph_capture_sizes=[512,504,496,488,480,472,464,456,448,440,432,424,416,408,400,392,384,376,368,360,352,344,336,328,320,312,304,296,288,280,272,264,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],
        max_capture_size=512
    )

    # Model arguments as a proper dict
    model_args = {
        "pretrained": f"/disk/models/{args.model}",
        "tensor_parallel_size": 1,
        "add_bos_token": True,
        "dtype": "float16",
        "quantization": "nestedfp",
        "compilation_config": compilation_config,
        "trust_remote_code": True
    }

    # Call lm_eval
    results = evaluator.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=[f"{args.task}"],
        batch_size="auto",
        confirm_run_unsafe_code=True
    )

    # Print or save results
    print(results)

if __name__ == "__main__":
    main()