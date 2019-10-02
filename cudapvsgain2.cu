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

typedef struct _cudapvsgain2 {
  OPDS    h;
  PVSDAT  *fout;
  PVSDAT  *fa;
  MYFLT   *kgain;
  // float* deviceFrame;   // pointer to device memory (NEEDED NO MORE)
  int gridSize;   // number of blocks in the grid (1D)
  int blockSize;   // number of threads in one block (1D)
  uint32  lastframe;
} CUDAPVSGAIN2;

// kernel for scaling PV amplitudes
__global__ void applygain(float* output, float* input, MYFLT g, int length) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = i<<1; 
  if(j < length) {
    output[j] = (float) input[j] * g; 
    output[j+1] = input[j+1]; 
  }     
} 

static int free_device(CSOUND* csound, void* pp){
  CUDAPVSGAIN2* p = (CUDAPVSGAIN2*) pp;
  cudaFree(p->fout->frame.auxp);
  return OK;
} 

// NO MORE ERRORS EXPECTED!
/*
static void handleCudaError (CSOUND *csound, cudaError_t error) {
  if (error!= cudaSuccess) {
    csound->Message(csound, "%s in %s at line %d\n", cudaGetErrorString(error),__FILE__,
       __LINE__);
    exit(EXIT_FAILURE);
  }
} 
*/

static int cudapvsgain2set(CSOUND *csound, CUDAPVSGAIN2 *p){

  int32 N = p->fa->N;
  int size = (N+2) * sizeof(float);
  int maxBlockDim;
  int SMcount;
  int totNumThreads = (N+2)/2;
  // cudaError_t error;

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp,0);
  maxBlockDim = deviceProp.maxThreadsPerBlock;
  SMcount = deviceProp.multiProcessorCount;
  csound->Message(csound, "cudapvsgain2 running on device %s (capability %d.%d)\n", deviceProp.name,
     deviceProp.major, deviceProp.minor);
   
  // error = cudaMalloc(&p->deviceFrame, size);
  // handleCudaError(csound, error);
  // cudaMemcpy(p->deviceFrame,p->fa->frame.auxp,size,cudaMemcpyHostToDevice);
    
  p->fout->sliding = 0;
    
  if (p->fout->frame.auxp == NULL || p->fout->frame.size < size)
    AuxCudaAlloc(size, &p->fout->frame);
    
  p->blockSize = (((totNumThreads/SMcount)/32)+1)*32;
  if (p->blockSize > maxBlockDim) p->blockSize = maxBlockDim;
  p->gridSize = totNumThreads / p->blockSize + 1;  
  p->fout->N = N;
  p->fout->overlap = p->fa->overlap;
  p->fout->winsize = p->fa->winsize;
  p->fout->wintype = p->fa->wintype;
  p->fout->format = p->fa->format;
  p->fout->framecount = 1;
  p->lastframe = 0;
  /*if (UNLIKELY(!(p->fout->format == PVS_AMP_FREQ) ||
                 (p->fout->format == PVS_AMP_PHASE)))
    return csound->InitError(csound, Str("cudapvsgain2: signal format "
                                           "must be amp-phase or amp-freq."));*/
  
  csound->RegisterDeinitCallback(csound, p, free_device);
  
  return OK;
}

static int cudapvsgain2(CSOUND *csound, CUDAPVSGAIN2 *p)
{
  int32   framelength = p->fa->N + 2;
  // int size = (int) framelength * sizeof(float);
  MYFLT gain = *p->kgain;
  float* fo = (float*) p->fout->frame.auxp;
  float* fi = (float*) p->fa->frame.auxp;

  if (p->lastframe < p->fa->framecount) {
    // cudaMemcpy(p->deviceFrame, p->fa->frame.auxp, size, cudaMemcpyHostToDevice); 
    applygain<<<p->gridSize,p->blockSize>>>(fo, fi, gain, framelength);   // KERNEL LAUNCH
    // cudaMemcpy(p->fout->frame.auxp, p->deviceFrame, size, cudaMemcpyDeviceToHost);
    p->fout->framecount = p->fa->framecount;
    p->lastframe = p->fout->framecount;
  }

  return OK;
}


static OENTRY localops[] = {
  {"cudapvsgain2", sizeof(CUDAPVSGAIN2), 0, 3, "f", "fk",
                               (SUBR) cudapvsgain2set, (SUBR) cudapvsgain2, NULL}
};

extern "C" {
  LINKAGE
}
