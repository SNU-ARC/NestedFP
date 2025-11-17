#!/bin/bash

set -e # Exit on error

# 1. Create environment
conda env create -f nestedfp.yml

# 2. Install vLLM 0.8.5 precompiled version
#    Clone vLLM into a temporary folder, then copy only the .git directory
mkdir -p tmp && cd tmp
git clone https://github.com/vllm-project/vllm.git
cd ..

cp -r tmp/vllm/.git vllm/
rm -rf tmp

cd vllm
git add .
git commit -m "nestedfp"
git branch install
git checkout install
git reset --hard f192ca90e6e8ab7b1b0015040e521c5374f5c812

# Install the precompiled vLLM binary
VLLM_USE_PRECOMPILED=1 pip install --editable .

# Return to the main branch with NestedFP changes
git checkout main

# 3. Install NestedFP kernels
cd ../nestedfp
./run.sh
