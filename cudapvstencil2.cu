// -*- c++ -*-
/*
  experimental cuda opcodes

  (c) Andrea Crespi, Victor Lazzarini, 2016

  This file is part of Csound.

  The Csound Library is free software; you can redistribute it
  and/or modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  Csound is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with Csound; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
  02111-1307 USA
*/

#include <csdl.h>
#include <pstream.h>

static void AuxCudaAlloc(int size, AUXCH *p){
  float *mem;
  cudaMalloc(&mem, size);
  cudaMemset(mem, 0, size);  
  p->auxp = mem;
  p->size = size;
}

#include <thrust/replace.h>
#include <thrust/transform.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

typedef struct _cudapvstencil2 {
    OPDS    h;
    PVSDAT  *fout;
    PVSDAT  *fin;
    MYFLT   *kgain;
    MYFLT   *klevel;
    MYFLT   *ifn;
    FUNC    *func;
    MYFLT*  devStencil;  // device memory pointer (function table to use as stencil)  
    int     gridSize;    // number of blocks in the grid (1D)
    int     blockSize;   // number of threads in one block (1D)
    uint32  lastframe;
} CUDAPVSTENCIL2;

struct _is_less_than_zero {
  __host__ __device__
  bool operator()(float x) {return x < 0.0f;}
};

__global__ void naivekernel(float* output, float* input, MYFLT* stencil, float level, float gain, int length){
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = i<<1;
  if (i < length) {  
    if (input[j] < (((float) stencil[i])*level)) {
      output[j] = input[j] * gain;
      output[j+1] = input[j+1];
    }
    else {
      output[j] = input[j];
      output[j+1] = input[j+1];
    }
  }
}

static int free_device(CSOUND* csound, void* pp){
  CUDAPVSTENCIL2* p = (CUDAPVSTENCIL2*) pp;
  cudaFree(p->fout->frame.auxp);
  cudaFree(p->devStencil);
  return OK;
} 

static void handleCudaError (CSOUND *csound, cudaError_t error) {
  if (error != cudaSuccess) {
    csound->Message(csound, "%s in %s at line %d\n", cudaGetErrorString(error),__FILE__,
       __LINE__);
    exit(EXIT_FAILURE);
  }
} 

static int cudapvstencil2set(CSOUND *csound, CUDAPVSTENCIL2 *p)
{
  int32    N = p->fin->N;
  int framelength = N+2;
  int chans = framelength>>1;
  int size = framelength*sizeof(float);
  int stencilSize = chans*sizeof(MYFLT);

  int maxBlockDim;
  int SMcount;
  int totNumThreads = chans;
  cudaError_t error;

  // get info about device
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp,0);
  maxBlockDim = deviceProp.maxThreadsPerBlock;
  SMcount = deviceProp.multiProcessorCount;
  csound->Message(csound, "cudapvstencil2 running on device %s (capability %d.%d)\n", deviceProp.name,
     deviceProp.major, deviceProp.minor);

  // device memory allocation (stencil)
  error = cudaMalloc(&p->devStencil, stencilSize);
  handleCudaError(csound, error);

  p->fout->sliding = 0;

  p->blockSize = (((totNumThreads/SMcount)/32)+1)*32;
  if (p->blockSize > maxBlockDim) p->blockSize = maxBlockDim;
  p->gridSize = totNumThreads / p->blockSize + 1;
  p->fout->N = N;
  p->fout->overlap = p->fin->overlap;
  p->fout->winsize = p->fin->winsize;
  p->fout->wintype = p->fin->wintype;
  p->fout->format = p->fin->format;
  p->fout->framecount = 1;
  p->lastframe = 0;

  p->fout->NB = chans;
  
  if (p->fout->frame.auxp == NULL || p->fout->frame.size < sizeof(float) * (N + 2))
      AuxCudaAlloc(size, &p->fout->frame);

  p->func = csound->FTnp2Find(csound, p->ifn);
  if (p->func == NULL)
    return OK;

  if (UNLIKELY(p->func->flen + 1 < (unsigned int)chans))
    return csound->InitError(csound, Str("cudapvstencil2: ftable needs to equal "
                                           "the number of bins"));

  cudaMemcpy(p->devStencil, p->func->ftable, stencilSize, cudaMemcpyHostToDevice);

  thrust::device_ptr<MYFLT> dev_ptr = thrust::device_pointer_cast(p->devStencil);

  _is_less_than_zero pred;
  thrust::replace_if(thrust::device, dev_ptr, dev_ptr + chans, pred, (MYFLT) 0.0);
  
  cudaMemcpy(p->func->ftable, p->devStencil, stencilSize, cudaMemcpyDeviceToHost);

  csound->RegisterDeinitCallback(csound, p, free_device);

  return OK;
}

static int cudapvstencil2(CSOUND *csound, CUDAPVSTENCIL2 *p)
{
  int framelength = p->fin->N + 2;
  int chans = framelength>>1;
  float* fout = (float *) p->fout->frame.auxp;
  float* fin = (float *) p->fin->frame.auxp;
  float   g = fabsf((float)*p->kgain);
  float   level = fabsf((float)*p->klevel);

  if (UNLIKELY(fout == NULL)) goto err1;

  if (p->lastframe < p->fin->framecount) {

    naivekernel<<<p->gridSize,p->blockSize>>>(fout, fin, p->devStencil, level, g, chans);    
 
    p->fout->framecount = p->lastframe = p->fin->framecount;
  }

  return OK;
  err1:
    return csound->PerfError(csound, p->h.insdshead,
                             Str("cudapvstencil2: not initialised"));
}


static OENTRY localops[] = {
  {"cudapvstencil2", sizeof(CUDAPVSTENCIL2), TR, 3, "f", "fkki", (SUBR) cudapvstencil2set,
   (SUBR) cudapvstencil2}
};

extern "C" {
  LINKAGE
}

