CUDA_VISIBLE_DEVICES=4,5,6,7 python vllm_simple_server.py --model /home/ubuntu/disk/models/Llama-3.1-70B --port 8070 --tensor-parallel-size 4 --quantization nestedfp  &> nestedfp_server.log
# CUDA_VISIBLE_DEVICES=4,5,6,7 python vllm_simple_server.py --model /home/ubuntu/models/Llama-3.1-70B --port 8070 --tensor-parallel-size 4
