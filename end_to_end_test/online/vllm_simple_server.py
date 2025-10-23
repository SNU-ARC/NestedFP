import asyncio
import time
import uuid
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import argparse

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

# ===== Argument Parser =====
parser = argparse.ArgumentParser(description="vLLM Server with configurable options")
parser.add_argument("--model", type=str, 
                    default="/home/ubuntu/models/Llama-3.1-8B",
                    help="Model path (default: /home/ubuntu/models/Llama-3.1-8B)")
parser.add_argument("--port", type=int, 
                    default=8000,
                    help="Server port (default: 8000)")
parser.add_argument("--gpu", type=float, 
                    default=0.9,
                    help="GPU memory utilization (default: 0.9)")
parser.add_argument("--quantization", type=str, 
                    default=None,
                    choices=["nestedfp", "awq", "gptq", "fp8", None],
                    help="Quantization method (default: None for no quantization)")
parser.add_argument("--tensor-parallel-size", type=int,
                    default=1,
                    help="Tensor parallel size (default: 1)")
parser.add_argument("--max-model-len", type=int,
                    default=16384,
                    help="Maximum model length (default: 16384)")
parser.add_argument("--max-num-batched-tokens", type=int,
                    default=16384,
                    help="Maximum number of batched tokens (default: 2048)")

parser.add_argument("--enforce-eager", action="store_true",
                    help="Disable CUDA graph and use eager mode")

args = parser.parse_args()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Engine Configuration =====
print(f"\n{'='*80}")
print("vLLM Server Configuration")
print(f"{'='*80}")
print(f"Model: {args.model}")
print(f"Port: {args.port}")
print(f"Quantization: {args.quantization if args.quantization else 'None (FP16)'}")
print(f"GPU Memory Utilization: {args.gpu}")
print(f"Tensor Parallel Size: {args.tensor_parallel_size}")
print(f"Max Model Length: {args.max_model_len}")
print(f"Max Num Batched Tokens: {args.max_num_batched_tokens}")
print(f"{'='*80}\n")

# Engine args 동적 생성
max_batch = 8192
CAPTURE_SIZES = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
capture_sizes = [s for s in CAPTURE_SIZES if s <= max_batch]

engine_args_dict = {
    "model": args.model,
    "tensor_parallel_size": args.tensor_parallel_size,
    "dtype": "float16",
    "enable_prefix_caching": False,
    "gpu_memory_utilization": args.gpu,
    "max_num_batched_tokens": args.max_num_batched_tokens,
    "max_model_len": args.max_model_len,
    # "max_num_seqs": 512,
}

if not args.enforce_eager:
    engine_args_dict["compilation_config"] = {
        "level": 3,
        "custom_ops": ["none"],
        "splitting_ops": [
            "vllm.unified_attention",
            "vllm.unified_attention_with_output",
        ],
        "full_cuda_graph": True,
        "use_inductor": True,
        "compile_sizes": capture_sizes,
        "use_cudagraph": True,
        "cudagraph_num_of_warmups": 2,
        "cudagraph_capture_sizes": capture_sizes,
        "max_capture_size": max_batch,
    }
else:
    engine_args_dict["enforce_eager"] = True

# quantization이 지정된 경우에만 추가
if args.quantization:
    engine_args_dict["quantization"] = args.quantization

# quantization이 지정된 경우에만 추가
if args.quantization:
    engine_args_dict["quantization"] = args.quantization

engine_args = AsyncEngineArgs(**engine_args_dict)
engine = AsyncLLMEngine.from_engine_args(engine_args)

@app.post("/v1/completions")
async def completions(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 128)
        stream = data.get("stream", False)
        temperature = data.get("temperature", 0.0)
        top_p = data.get("top_p", 1.0)
        stop = data.get("stop", None)
        echo = data.get("echo", False)
        model = args.model  # 실제 모델명 사용

        request_id = str(uuid.uuid4())
        created_time = int(time.time())
        start_time = time.time()

        # 요청 시작 로깅
        print(f"[{request_id[:8]}] Request started")

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            ignore_eos=True
        )

        if not stream:
            full_text = ""
            num_prompt_tokens = 0
            num_completion_tokens = 0

            async for output in engine.generate(prompt, sampling_params, request_id=request_id):
                full_text = output.outputs[0].text
                num_prompt_tokens = output.prompt_token_ids.__len__()
                num_completion_tokens = output.outputs[0].token_ids.__len__()

            elapsed_time = time.time() - start_time
            # 요청 완료 로깅
            print(f"[{request_id[:8]}] Request completed in {elapsed_time:.2f}s")

            return {
                "id": f"cmpl-{request_id[:8]}",
                "object": "text_completion",
                "created": created_time,
                "model": model,
                "choices": [{
                    "text": full_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": num_prompt_tokens,
                    "completion_tokens": num_completion_tokens,
                    "total_tokens": num_prompt_tokens + num_completion_tokens,
                },
            }
        else:
            async def stream_response():
                async for output in engine.generate(prompt, sampling_params, request_id=request_id):
                    for token in output.outputs:
                        chunk = {
                            "choices": [{
                                "text": token.text,
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": None,
                                # 🔄 기존 필드들
                                "iteration_total": getattr(output, 'iteration_total', None),
                                "iteration_timestamp": getattr(output, 'iteration_timestamp', None),
                                "kv_cache_usage": getattr(output, 'kv_cache_usage', None),
                                "kv_cache_usage_gb": getattr(output, 'kv_cache_usage_gb', None),
                                "kv_cache_total_capacity": getattr(output, 'kv_cache_total_capacity', None),
                                "num_prefill": getattr(output, 'num_prefill', None),
                                "num_decode": getattr(output, 'num_decode', None),
                                
                                # 🆕 새로운 스케줄링 필드들 추가
                                "total_scheduled_requests": getattr(output, 'total_scheduled_requests', None),
                                "total_scheduled_tokens": getattr(output, 'total_scheduled_tokens', None),
                                "prefill_requests": getattr(output, 'prefill_requests', None),
                                "decode_requests": getattr(output, 'decode_requests', None),
                                "prefill_tokens": getattr(output, 'prefill_tokens', None),
                                "decode_tokens": getattr(output, 'decode_tokens', None),
                                "request_details": getattr(output, 'request_details', []),
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                
                elapsed_time = time.time() - start_time
                # 스트림 완료 로깅
                print(f"[{request_id[:8]}] Stream completed in {elapsed_time:.2f}s")
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_response(), media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)