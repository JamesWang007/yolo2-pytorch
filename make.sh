#!/usr/bin/env bash

#PYTHON=python
PYTHON=python3

NVCC=nvcc

cd utils
${PYTHON} build.py build_ext --inplace
cd ../

cd layers/reorg/src
echo "Compiling reorg layer kernels by nvcc..."
${NVCC} -c -o reorg_cuda_kernel.cu.o reorg_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_61
cd ../
${PYTHON} build.py
cd ../

cd roi_pooling/src/cuda
echo "Compiling roi_pooling kernels by nvcc..."
${NVCC} -c -o roi_pooling_kernel.cu.o roi_pooling_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_61
cd ../../
${PYTHON} build.py
cd ../
