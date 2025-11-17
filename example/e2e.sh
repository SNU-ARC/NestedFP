#!/usr/bin/env bash
set -euo pipefail

PORT=8000
MAX_TOKENS=8192

run_server_and_client() {
    local model_path="$1"
    local extra_server_args="$2"
    local extra_client_args="$3"

    local model_name
    model_name=$(basename "$model_path")

    # Decide mode name based on whether nestedfp is enabled
    local mode_name="fp16"
    if [[ "$extra_server_args" == *"nestedfp"* ]] || [[ "$extra_client_args" == *"nestedfp"* ]]; then
        mode_name="nestedfp"
    fi

    # Log file names
    local server_log_file="server_${model_name}_${mode_name}.log"
    local client_log_file="client_${model_name}_${mode_name}.log"

    echo
    echo "============================================="
    echo "Model:       $model_name"
    echo "Mode:        $mode_name"
    echo "Server args: $extra_server_args"
    echo "Client args: $extra_client_args"
    echo "Server log:  $server_log_file"
    echo "Client log:  $client_log_file"
    echo "============================================="

    # Clean up old logs
    rm -f "$server_log_file" "$client_log_file"
    touch "$server_log_file" "$client_log_file"

    # Launch server in the background (log to server_log_file)
    python scripts/vllm_simple_server.py \
        --model "$model_path" \
        --max-num-batched-tokens "$MAX_TOKENS" \
        --port "$PORT" \
        $extra_server_args \
        >"$server_log_file" 2>&1 &

    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"
    echo "Waiting for server to become ready..."

    # Wait until "Application startup complete" appears in the server log
    while ! grep -q "Application startup complete" "$server_log_file"; do
        # Check if server died unexpectedly
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: server process exited unexpectedly. Check $server_log_file"
            return 1
        fi
        sleep 1
    done

    echo "Server is ready. Running client..."

    # Run client and log output to client_log_file
    # If client fails, do NOT abort whole script (because of `set -e`)
    if ! python scripts/vllm_simple_client.py \
        --model "$model_path" \
        --api-url "http://0.0.0.0:${PORT}/v1/completions" \
        --test-mode throughput \
        $extra_client_args \
        >"$client_log_file" 2>&1; then
        echo "WARNING: client exited with non-zero status. See $client_log_file"
    fi

    echo "Client finished. Stopping server (PID=$SERVER_PID)..."

    # Try graceful shutdown of server
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    # Also try to terminate direct children of the server (if any)
    pkill -TERM -P "$SERVER_PID" 2>/dev/null || true
    sleep 5

    # If server is still alive, force kill it
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server still alive. Force killing PID=$SERVER_PID..."
        kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi

    # Also force kill any remaining direct children
    pkill -KILL -P "$SERVER_PID" 2>/dev/null || true

    # Reap zombies
    wait "$SERVER_PID" 2>/dev/null || true

    echo "Done for model: $model_name (mode: $mode_name)."
    echo "  - Server log: $server_log_file"
    echo "  - Client log: $client_log_file"
}

#####################
# Actual run sequence
#####################

# Llama-3.1-8B
run_server_and_client \
    "/home/snu_arclab_2nd/models/Llama-3.1-8B" \
    "" \
    ""

run_server_and_client \
    "/home/snu_arclab_2nd/models/Llama-3.1-8B" \
    "--quantization nestedfp" \
    "--nestedfp"

# Mistral-Nemo-Base-2407
run_server_and_client \
    "/home/snu_arclab_2nd/models/Mistral-Nemo-Base-2407" \
    "" \
    ""

run_server_and_client \
    "/home/snu_arclab_2nd/models/Mistral-Nemo-Base-2407" \
    "--quantization nestedfp" \
    "--nestedfp"

# phi-4
run_server_and_client \
    "/home/snu_arclab_2nd/models/phi-4" \
    "" \
    ""

run_server_and_client \
    "/home/snu_arclab_2nd/models/phi-4" \
    "--quantization nestedfp" \
    "--nestedfp"

# Mistral-Small-24B-Base-2501
run_server_and_client \
    "/home/snu_arclab_2nd/models/Mistral-Small-24B-Base-2501" \
    "" \
    ""

run_server_and_client \
    "/home/snu_arclab_2nd/models/Mistral-Small-24B-Base-2501" \
    "--quantization nestedfp" \
    "--nestedfp"
