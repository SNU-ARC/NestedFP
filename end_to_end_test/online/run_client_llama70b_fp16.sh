python vllm_simple_client.py --model /home/ubuntu/disk/models/Llama-3.1-70B --api-url http://0.0.0.0:8100/v1/completions --num-requests 1000 --middle-ratio 0.7 --test-mode trace
python analysis_benchmark_request.py
