#!/bin/bash

rm -rf build/ quant.egg-info/ dist/
TORCH_CUDA_ARCH_LIST="9.0" python setup.py install > compile.log
