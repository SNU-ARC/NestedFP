python scripts/vllm_simple_server.py --model /home/snu_arclab_2nd/models/Llama-3.1-8B --max-num-batched-tokens 8192 --port 8000 
python scripts/vllm_simple_client.py --model /home/snu_arclab_2nd/models/Llama-3.1-8B --api-url http://0.0.0.0:8000/v1/completions --test-mode throughput

python scripts/vllm_simple_server.py --model /home/snu_arclab_2nd/models/Llama-3.1-8B --max-num-batched-tokens 8192 --port 8000 --quantization nestedfp 
python scripts/vllm_simple_client.py --model /home/snu_arclab_2nd/models/Llama-3.1-8B --api-url http://0.0.0.0:8000/v1/completions --test-mode throughput --nestedfp



python scripts/vllm_simple_server.py --model /home/snu_arclab_2nd/models/Mistral-Nemo-Base-2407 --max-num-batched-tokens 8192 --port 8000 
python scripts/vllm_simple_client.py --model /home/snu_arclab_2nd/models/Mistral-Nemo-Base-2407 --api-url http://0.0.0.0:8000/v1/completions --test-mode throughput

python scripts/vllm_simple_server.py --model /home/snu_arclab_2nd/models/Mistral-Nemo-Base-2407 --max-num-batched-tokens 8192 --port 8000 --quantization nestedfp 
python scripts/vllm_simple_client.py --model /home/snu_arclab_2nd/models/Mistral-Nemo-Base-2407 --api-url http://0.0.0.0:8000/v1/completions --test-mode throughput --nestedfp



python scripts/vllm_simple_server.py --model /home/snu_arclab_2nd/models/phi-4 --max-num-batched-tokens 8192 --port 8000 
python scripts/vllm_simple_client.py --model /home/snu_arclab_2nd/models/phi-4 --api-url http://0.0.0.0:8000/v1/completions --test-mode throughput

python scripts/vllm_simple_server.py --model /home/snu_arclab_2nd/models/phi-4 --max-num-batched-tokens 8192 --port 8000 --quantization nestedfp 
python scripts/vllm_simple_client.py --model /home/snu_arclab_2nd/models/phi-4 --api-url http://0.0.0.0:8000/v1/completions --test-mode throughput --nestedfp



python scripts/vllm_simple_server.py --model /home/snu_arclab_2nd/models/Mistral-Small-24B-Base-2501 --max-num-batched-tokens 8192 --port 8000 
python scripts/vllm_simple_client.py --model /home/snu_arclab_2nd/models/Mistral-Small-24B-Base-2501 --api-url http://0.0.0.0:8000/v1/completions --test-mode throughput

python scripts/vllm_simple_server.py --model /home/snu_arclab_2nd/models/Mistral-Small-24B-Base-2501 --max-num-batched-tokens 8192 --port 8000 --quantization nestedfp 
python scripts/vllm_simple_client.py --model /home/snu_arclab_2nd/models/Mistral-Small-24B-Base-2501 --api-url http://0.0.0.0:8000/v1/completions --test-mode throughput --nestedfp
