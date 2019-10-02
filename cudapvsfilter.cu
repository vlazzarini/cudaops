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

typedef struct _cudapvsfilter {
  OPDS    h;
  PVSDAT  *fout;
  PVSDAT  *fin;
  PVSDAT  *fmask;
  MYFLT   *kdepth;
  MYFLT   *igain; 
  float*  deviceInput;   // pointer to device memory (input frame)
  float*  deviceOutput;   // pointer to device memory (output frame)
  float*  deviceMask;   // pointer to device memory (mask frame)
  int  gridSize;   // number of blocks in the grid (1D)
  int blockSize;   // number of threads in one block (1D)
  uint32  lastframe;
} CUDAPVSFILTER;

// kernel for filtering in the frequency domain, given a spectral mask
__global__ void filter(float* input, float* output, float* mask, MYFLT wet, MYFLT dry, float g, int length) {
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = i<<1;
  if (j < length) {
    output[j] = (float) (input[j]*(dry+mask[j]*wet))*g;
    output[j+1] = input[j+1];
  }
} 

static int fsigs_equal(const PVSDAT *f1, const PVSDAT *f2)
{
    if (
        (f1->sliding == f2->sliding) &&
        (f1->overlap == f2->overlap) &&
        (f1->winsize == f2->winsize) &&
        (f1->wintype == f2->wintype) &&     /* harsh, maybe... */
        (f1->N == f2->N) &&
        (f1->format == f2->format))

      return 1;
    return 0;
}

static int free_device(CSOUND* csound, void* pp){
  CUDAPVSFILTER* p = (CUDAPVSFILTER*) pp;
  cudaFree(p->deviceInput);
  cudaFree(p->deviceOutput);
  cudaFree(p->deviceMask);
  return OK;
} 

static void handleCudaError (CSOUND *csound, cudaError_t error) {
  if (error!= cudaSuccess) {
    csound->Message(csound, "%s in %s at line %d\n", cudaGetErrorString(error),__FILE__,
       __LINE__);
    exit(EXIT_FAILURE);
  }
} 

static int cudapvsfilterset(CSOUND *csound, CUDAPVSFILTER *p)
{
  int N = p->fin->N;
  int size = (N+2) * sizeof(float);
  int maxBlockDim;
  int SMcount;
  int totNumThreads = (N+2)>>1;   // TO BE MODIFIED, MAYBE NOT
  cudaError_t error;

  // get info about device
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp,0);
  maxBlockDim = deviceProp.maxThreadsPerBlock;
  SMcount = deviceProp.multiProcessorCount;
  csound->Message(csound, "cudapvsfilter running on device %s (capability %d.%d)\n", deviceProp.name,
     deviceProp.major, deviceProp.minor);
  
  // device memory allocation (input data) 
  error = cudaMalloc(&p->deviceInput, size);
  handleCudaError(csound, error);
  
  // device memory allocation (input mask) 
  error = cudaMalloc(&p->deviceMask, size);
  handleCudaError(csound, error);

  // device memory allocation and set to zero (output data to be computed)
  error = cudaMalloc(&p->deviceOutput, size);
  handleCudaError(csound, error);
  cudaMemset(p->deviceOutput,0,size);

  if (UNLIKELY(p->fin == p->fout || p->fmask == p->fout))
    csound->Warning(csound, Str("Unsafe to have same fsig as in (or filter) and out"));
  if (UNLIKELY(!(p->fout->format == PVS_AMP_FREQ) || (p->fout->format == PVS_AMP_PHASE)))
    return csound->InitError(csound, Str("cudapvsfilter: signal format "
                                           "must be amp-phase or amp-freq."));
  p->fout->sliding = 0;
  
  if (p->fout->frame.auxp == NULL || p->fout->frame.size < sizeof(float) * (N + 2))
    csound->AuxAlloc(csound, sizeof(float) * (N + 2), &p->fout->frame);
  
  p->blockSize = (((totNumThreads/SMcount)/32)+1)*32;
  if (p->blockSize > maxBlockDim) p->blockSize = maxBlockDim;
  p->gridSize = totNumThreads / p->blockSize + 1;  
  p->fout->N = N;
  p->fout->overlap = p->fin->overlap;
  p->fout->winsize = p->fin->winsize;
  p->fout->wintype = p->fin->wintype;
  p->fout->format = p->fin->format;
  p->fout->framecount = 1;
  *p->igain = 1.0;
  p->lastframe = 0;
  
  csound->RegisterDeinitCallback(csound, p, free_device);

  return OK;
}

static int cudapvsfilter(CSOUND *csound, CUDAPVSFILTER *p) {
  int N = p->fout->N;
  int framelength = N+2;
  int size = framelength * sizeof(float);
  float* fout = (float*) p->fout->frame.auxp; 
  float* fin = (float*) p->fin->frame.auxp;
  float* fmask = (float*) p->fmask->frame.auxp;
  MYFLT depth = *p->kdepth;
  float gain = (float) *p->igain;
  MYFLT dirgain;

  if (UNLIKELY(fout == NULL)) goto err1;
  if (UNLIKELY(!fsigs_equal(p->fin,p->fmask))) goto err2;
  
  if (p->lastframe < p->fin->framecount) {
    
    cudaMemcpy(p->deviceInput,fin,size,cudaMemcpyHostToDevice);
    cudaMemcpy(p->deviceMask,fmask,size,cudaMemcpyHostToDevice);
    
    depth = depth >= 0 ? (depth <= 1 ? depth : 1) : FL(0.0);   // clip depth between zero and one
    dirgain = (1 - depth);
    
    // filtering on the GPU:
    filter<<<p->gridSize,p->blockSize>>>(p->deviceInput, p->deviceOutput, p->deviceMask, depth, dirgain, gain, framelength);
    
    cudaMemcpy(fout,p->deviceOutput,size,cudaMemcpyDeviceToHost);

    p->fout->framecount = p->lastframe = p->fin->framecount; 
  } 
 
  return OK;

  err1: return csound->PerfError(csound, p->h.insdshead, Str("cudapvsfilter: not initialised"));
  err2: return csound->PerfError(csound, p->h.insdshead, Str("cudapvsfilter: formats are different."));
}

static OENTRY localops[] = {
  {"cudapvsfilter", sizeof(CUDAPVSFILTER),0, 3, "f", "ffxp", (SUBR) cudapvsfilterset,
   (SUBR) cudapvsfilter},
};

extern "C" {
  LINKAGE
}

