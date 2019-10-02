#!/bin/sh
# use -Xptxas="-v" to check register usage and --maxrregcount 32 to limit it
echo "building cuda opcodes ..."
nvcc -O3 -shared -o libcudapvs2.dylib pvsops2.cu -use_fast_math -Xcompiler "-fPIC" -I../../debug/CsoundLib64.framework/Headers -arch=sm_50 -I/usr/local/cuda/include -L/usr/local/cuda/lib -I$HOME/include/csound -lcufft
echo "...done"
