import asyncio
import time
import uuid
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine_args = AsyncEngineArgs(
    # model="/disk/models/Llama-3.1-8B",
    model="/disk/models/Llama-3.1-8B-FP8-Dynamic",
    # model="/disk/models/Mistral-Small-24B-Base-2501",
    # model="/disk/models/Mistral-Small-24B-Base-2501-FP8-Dynamic",
    dtype="float16",
    enable_prefix_caching=False
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
        model = "mistral-24b"  # 정해진 모델 이름, 실제 모델명과 매칭해도 무방

        request_id = str(uuid.uuid4())
        created_time = int(time.time())

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
                first_chunk = True
                async for output in engine.generate(prompt, sampling_params, request_id=request_id):
                    for token in output.outputs:
                        chunk = {
                            "choices": [{
                                "text": token.text,
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_response(), media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
