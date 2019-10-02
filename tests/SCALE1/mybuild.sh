#!/bin/sh
# use -Xptxas="-v" to check register usage and --maxrregcount 32 to limit it
echo "building cuda opcodes ..."
nvcc -O3 -shared -o libcudapvscale.so ../../cudapvscale1.cu -Xcompiler "-fPIC" -I$HOME/include/csound -arch=sm_50 -I/usr/local/cuda/include -L/usr/local/cuda/lib -lcufft
echo "...done"
