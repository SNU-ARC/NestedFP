# python vllm_simple_client.py --duration-minutes 20 --middle-ratio 0.7
python vllm_simple_client.py --num-requests 100 --middle-ratio 0.7
# python vllm_simple_client.py --duration-minutes 20 --middle-ratio 0.7
python analysis_benchmark_request.py
# python analysis_benchmark_iteration.py
# python analysis_benchmark_budget.py

# ./cp_results.sh default_exp