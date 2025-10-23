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

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, help="port", default=8000)
parser.add_argument("--gpu", type=float, help="gpu-memory-utilization", default=0.9)

args = parser.parse_args()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine_args = AsyncEngineArgs(
    # model="Qwen/Qwen2.5-7B",
    model = "/home/ubuntu/models/Llama-3.1-70B",
    tensor_parallel_size=4,
    dtype="float16",
    quantization="nestedfp",
    enable_prefix_caching=False,
    gpu_memory_utilization=args.gpu,
    max_num_batched_tokens=2048,
    max_model_len= 16384,
    enforce_eager= True,
)
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
        model = "unknown"

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