CUDA_VISIBLE_DEVICES=0,1,2,3 python vllm_simple_server.py --model /home/ubuntu/disk/models/Llama-3.1-70B --port 8100 --tensor-parallel-size 4 &> fp16_server.log
