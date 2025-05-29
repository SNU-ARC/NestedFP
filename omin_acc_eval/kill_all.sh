#!/bin/bash

echo "=== Killing all evaluate_vllm_nested_fp8.sh related processes ==="

# 1. 스크립트 자체 프로세스 찾기 및 종료
echo "1. Finding and killing script processes..."
pids=$(pgrep -f "evaluate_vllm_nested_fp8.sh" 2>/dev/null)
if [ -n "$pids" ]; then
    echo "Found script processes: $pids"
    kill -TERM $pids 2>/dev/null
    sleep 2
    # 강제 종료가 필요한 경우
    pids=$(pgrep -f "evaluate_vllm_nested_fp8.sh" 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "Force killing remaining script processes: $pids"
        kill -KILL $pids 2>/dev/null
    fi
else
    echo "No script processes found"
fi

# 2. lm_eval 프로세스 찾기 및 종료
echo ""
echo "2. Finding and killing lm_eval processes..."
pids=$(pgrep -f "lm_eval" 2>/dev/null)
if [ -n "$pids" ]; then
    echo "Found lm_eval processes: $pids"
    kill -TERM $pids 2>/dev/null
    sleep 3
    # 강제 종료가 필요한 경우
    pids=$(pgrep -f "lm_eval" 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "Force killing remaining lm_eval processes: $pids"
        kill -KILL $pids 2>/dev/null
    fi
else
    echo "No lm_eval processes found"
fi

# 3. vLLM 관련 Python 프로세스 찾기 및 종료
echo ""
echo "3. Finding and killing vLLM Python processes..."
pids=$(pgrep -f "python.*vllm" 2>/dev/null)
if [ -n "$pids" ]; then
    echo "Found vLLM Python processes: $pids"
    kill -TERM $pids 2>/dev/null
    sleep 3
    # 강제 종료가 필요한 경우
    pids=$(pgrep -f "python.*vllm" 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "Force killing remaining vLLM Python processes: $pids"
        kill -KILL $pids 2>/dev/null
    fi
else
    echo "No vLLM Python processes found"
fi

# 4. 모델 관련 Python 프로세스 찾기 (Llama, Mistral, phi 등)
echo ""
echo "4. Finding and killing model-related Python processes..."
model_patterns=("Llama" "Mistral" "phi-4" "pretrained")
for pattern in "${model_patterns[@]}"; do
    pids=$(pgrep -f "python.*$pattern" 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "Found $pattern related processes: $pids"
        kill -TERM $pids 2>/dev/null
    fi
done

sleep 3

# 강제 종료가 필요한 경우
for pattern in "${model_patterns[@]}"; do
    pids=$(pgrep -f "python.*$pattern" 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "Force killing remaining $pattern processes: $pids"
        kill -KILL $pids 2>/dev/null
    fi
done

# 5. GPU 메모리 사용 중인 Python 프로세스 확인 및 정리
echo ""
echo "5. Checking GPU memory usage..."
if command -v nvidia-smi &> /dev/null; then
    echo "Current GPU processes:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | while read line; do
        if [[ "$line" == *"python"* ]]; then
            pid=$(echo "$line" | cut -d',' -f1 | tr -d ' ')
            echo "GPU Python process found: PID $pid - $line"
            # 선택적으로 이 프로세스들도 종료할 수 있습니다
            # kill -TERM $pid 2>/dev/null
        fi
    done
else
    echo "nvidia-smi not available, skipping GPU process check"
fi

# 6. 최종 확인
echo ""
echo "6. Final verification..."
remaining_script=$(pgrep -f "evaluate_vllm_nested_fp8.sh" 2>/dev/null | wc -l)
remaining_lmeval=$(pgrep -f "lm_eval" 2>/dev/null | wc -l)
remaining_vllm=$(pgrep -f "python.*vllm" 2>/dev/null | wc -l)

echo "Remaining processes:"
echo "  - Script processes: $remaining_script"
echo "  - lm_eval processes: $remaining_lmeval"
echo "  - vLLM Python processes: $remaining_vllm"

if [ $remaining_script -eq 0 ] && [ $remaining_lmeval -eq 0 ] && [ $remaining_vllm -eq 0 ]; then
    echo ""
    echo "✅ All evaluate_vllm_nested_fp8.sh related processes have been terminated!"
else
    echo ""
    echo "⚠️  Some processes may still be running. You might need to:"
    echo "   - Wait a bit longer for graceful shutdown"
    echo "   - Use 'kill -KILL <pid>' for stubborn processes"
    echo "   - Check GPU processes with 'nvidia-smi'"
fi

echo ""
echo "=== Process cleanup completed ==="