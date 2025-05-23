# VLLM_USE_V1=1 VLLM_FLASH_ATTN_VERSION=3  python test_llama8b_fp8_custom.py
# rm -rf /home/snu_arclab_2nd/.cache/vllm/torch_compile_cache/
# VLLM_USE_V1=1 VLLM_FLASH_ATTN_VERSION=3  python test_mistralnemo_fp8_custom.py
# rm -rf /home/snu_arclab_2nd/.cache/vllm/torch_compile_cache/
VLLM_USE_V1=1 VLLM_FLASH_ATTN_VERSION=3  python test_mistralsmall_fp8_custom.py
# rm -rf /home/snu_arclab_2nd/.cache/vllm/torch_compile_cache/
# VLLM_USE_V1=1 VLLM_FLASH_ATTN_VERSION=3  python test_phi4_fp8_custom.py
