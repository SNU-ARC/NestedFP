#!/usr/bin/env bash
# mv_results.sh

# Usage: ./mv_results.sh [exp_name]
# default exp_name=new_exp
EXP_NAME=${1:-new_exp}
DEST_DIR="/disk2/NestedFP_Results/${EXP_NAME}"

# 대상 디렉터리 생성
mkdir -p "${DEST_DIR}"

# 복사할 파일들
cp ./analysis_*.py    "${DEST_DIR}/"
cp ./*.json           "${DEST_DIR}/"
cp ./server.log       "${DEST_DIR}/"

echo "✅ 결과를 ${DEST_DIR} 에 복사했습니다."
