#!/bin/bash

# Trace Processor Runner Script

# Default input/output path
DEFAULT_INPUT="/disk/dualfp_vllm_test/end_to_end_test/online/trace/AzureLLMInferenceTrace_conv_1min.csv"
DEFAULT_OUTPUT="/disk/dualfp_vllm_test/end_to_end_test/online/trace/AzureLLMInferenceTrace_conv_10s.csv"

# 사용법 출력
usage() {
    echo "Usage: $0 [task] [options]"
    echo ""
    echo "Tasks:"
    echo "  clip_day [start_time] [end_time] [input_file] [output_file]"
    echo "      Example (default path): $0 clip_day '2024-05-14 00:00:00' '2024-05-14 23:59:59'"
    echo "      Example (custom path): $0 clip_day '2024-05-14 00:00:00' '2024-05-14 23:59:59' input.csv output.csv"
    echo ""
    echo "  extend_rate [n_copies] [time_offset_us] [input_file] [output_file]"
    echo "      Example (default path): $0 extend_rate 10 10"
    echo "      Example (custom path): $0 extend_rate 10 10 input.csv output.csv"
    echo ""
    echo "  clip_rows [n_rows] [input_file] [output_file]"
    echo "      Example (default path): $0 clip_rows 2000"
    echo "      Example (custom path): $0 clip_rows 2000 input.csv output.csv"
    echo ""
    exit 1
}

# 인자 최소 개수 체크
if [ "$#" -lt 1 ]; then
    usage
fi

TASK=$1
shift  # task를 제외한 나머지 인자만 남기기

# task별로 분기
if [ "$TASK" == "clip_day" ]; then
    if [ "$#" -lt 2 ]; then
        echo "Error: clip_day requires at least [start_time] [end_time]."
        usage
    fi
    START_TIME=$1
    END_TIME=$2
    INPUT_FILE=${3:-$DEFAULT_INPUT}
    OUTPUT_FILE=${4:-$DEFAULT_OUTPUT}

    python trace_processor.py \
        --task clip_day \
        --input "$INPUT_FILE" \
        --output "$OUTPUT_FILE" \
        --start_time "$START_TIME" \
        --end_time "$END_TIME"

elif [ "$TASK" == "extend_rate" ]; then
    if [ "$#" -lt 2 ]; then
        echo "Error: extend_rate requires at least [n_copies] [time_offset_us]."
        usage
    fi
    N_COPIES=$1
    TIME_OFFSET_US=$2
    INPUT_FILE=${3:-$DEFAULT_INPUT}
    OUTPUT_FILE=${4:-$DEFAULT_OUTPUT}

    python trace_processor.py \
        --task extend_rate \
        --input "$INPUT_FILE" \
        --output "$OUTPUT_FILE" \
        --n_copies "$N_COPIES" \
        --time_offset_us "$TIME_OFFSET_US"

elif [ "$TASK" == "clip_rows" ]; then
    if [ "$#" -lt 1 ]; then
        echo "Error: clip_rows requires at least [n_rows]."
        usage
    fi
    N_ROWS=$1
    INPUT_FILE=${2:-$DEFAULT_INPUT}
    OUTPUT_FILE=${3:-$DEFAULT_OUTPUT}

    python trace_processor.py \
        --task clip_rows \
        --input "$INPUT_FILE" \
        --output "$OUTPUT_FILE" \
        --n_rows "$N_ROWS"

elif [ "$TASK" == "plot_request_rate" ]; then
    INPUT_FILE=${1:-$DEFAULT_INPUT}
    BIN_INTERVAL_MS=${2:-1000}         # 기본값: 1000ms (=1초)
    MOVING_AVG_WINDOW_S=${3:-10}        # 기본값: 10초 moving average

    python trace_processor.py \
        --task plot_request_rate \
        --input "$INPUT_FILE" \
        --bin_interval_ms "$BIN_INTERVAL_MS" \
        --moving_avg_window_s "$MOVING_AVG_WINDOW_S"

elif [ "$TASK" == "extend_high_rate" ]; then
    INPUT_FILE=${1:-$DEFAULT_INPUT}
    OUTPUT_FILE=${2:-$DEFAULT_OUTPUT}
    RATE_MULTIPLIER=${3:-3}        # 기본값 3배
    DURATION_SECONDS=${4:-}        # 기본은 기존 구간 길이와 같음

    python trace_processor.py \
        --task extend_high_rate \
        --input "$INPUT_FILE" \
        --output "$OUTPUT_FILE" \
        --rate_multiplier "$RATE_MULTIPLIER" \
        ${DURATION_SECONDS:+--duration_seconds "$DURATION_SECONDS"}


else
    echo "Error: Unsupported task '$TASK'"
    usage
fi
