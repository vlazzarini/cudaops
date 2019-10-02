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

#include <thrust/replace.h>
#include <thrust/transform.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

typedef struct _cudapvstencil {
    OPDS    h;
    PVSDAT  *fout;
    PVSDAT  *fin;
    MYFLT   *kgain;
    MYFLT   *klevel;
    MYFLT   *ifn;
    FUNC    *func;
    float*  devFrame;    // device memory pointer (input/output frame)
    MYFLT*  devStencil;  // device memory pointer (function table to use as stencil)  
    int     gridSize;    // number of blocks in the grid (1D)
    int     blockSize;   // number of threads in one block (1D)
    uint32  lastframe;
} CUDAPVSTENCIL;

struct _is_less_than_zero {
  __host__ __device__
  bool operator()(float x) {return x < 0.0f;}
};

__global__ void naivekernel(float* frame, MYFLT* stencil, float level, float gain, int length){
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = i<<1;
  if (i < length) {  
    if (frame[j] < (((float) stencil[i])*level)) {
      frame[j] *= gain;
    }
  }
}

static int free_device(CSOUND* csound, void* pp){
  CUDAPVSTENCIL* p = (CUDAPVSTENCIL*) pp;
  cudaFree(p->devFrame);
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

static int cudapvstencilset(CSOUND *csound, CUDAPVSTENCIL *p)
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
  csound->Message(csound, "cudapvstencil running on device %s (capability %d.%d)\n", deviceProp.name,
     deviceProp.major, deviceProp.minor);

  // device memory allocation (PV frame)
  error = cudaMalloc(&p->devFrame, size);
  handleCudaError(csound, error);

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
      csound->AuxAlloc(csound, (N + 2) * sizeof(float), &p->fout->frame);

  if (UNLIKELY(!(p->fout->format == PVS_AMP_FREQ) || (p->fout->format == PVS_AMP_PHASE)))
      return csound->InitError(csound, Str("cudapvstencil: signal format "
                                               "must be amp-phase or amp-freq."));

  p->func = csound->FTnp2Find(csound, p->ifn);
  if (p->func == NULL)
    return OK;

  if (UNLIKELY(p->func->flen + 1 < (unsigned int)chans))
    return csound->InitError(csound, Str("cudapvstencil: ftable needs to equal "
                                           "the number of bins"));

  cudaMemcpy(p->devStencil, p->func->ftable, stencilSize, cudaMemcpyHostToDevice);

  thrust::device_ptr<MYFLT> dev_ptr = thrust::device_pointer_cast(p->devStencil);

  _is_less_than_zero pred;
  thrust::replace_if(thrust::device, dev_ptr, dev_ptr + chans, pred, (MYFLT) 0.0);
  
  cudaMemcpy(p->func->ftable, p->devStencil, stencilSize, cudaMemcpyDeviceToHost);

  csound->RegisterDeinitCallback(csound, p, free_device);

  return OK;
}

static int cudapvstencil(CSOUND *csound, CUDAPVSTENCIL *p)
{
  int framelength = p->fin->N + 2;
  int chans = framelength>>1;
  int size = framelength * sizeof(float);
  float* fout = (float *) p->fout->frame.auxp;
  float* fin = (float *) p->fin->frame.auxp;
  float   g = fabsf((float)*p->kgain);
  float   level = fabsf((float)*p->klevel);

  if (UNLIKELY(fout == NULL)) goto err1;

  if (p->lastframe < p->fin->framecount) {

    cudaMemcpy(p->devFrame, fin, size, cudaMemcpyHostToDevice); 

    naivekernel<<<p->gridSize,p->blockSize>>>(p->devFrame, p->devStencil, level, g, chans);

    cudaMemcpy(fout, p->devFrame, size, cudaMemcpyDeviceToHost);     
 
    p->fout->framecount = p->lastframe = p->fin->framecount;
  }

  return OK;
  err1:
    return csound->PerfError(csound, p->h.insdshead,
                             Str("cudapvstencil: not initialised"));
}


static OENTRY localops[] = {
  {"cudapvstencil", sizeof(CUDAPVSTENCIL), TR, 3, "f", "fkki", (SUBR) cudapvstencilset,
   (SUBR) cudapvstencil}
};

extern "C" {
  LINKAGE
}

