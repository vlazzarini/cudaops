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


typedef struct _cudapvsgain {
  OPDS    h;
  PVSDAT  *fout;
  PVSDAT  *fa;
  MYFLT   *kgain;
  float* deviceFrame;   // pointer to device memory
  int gridSize;   // number of blocks in the grid (1D)
  int blockSize;   // number of threads in one block (1D)
  uint32  lastframe;
} CUDAPVSGAIN;

// kernel for scaling PV amplitudes
__global__ void devicepvsgain (float* deviceFrame, MYFLT gain, int framesize) {
  int i = threadIdx.x + blockDim.x * blockIdx.x; 
  if(i < framesize>>1)
    deviceFrame[i<<1] *= gain;       
} 

static int free_device(CSOUND* csound, void* pp){
  CUDAPVSGAIN* p = (CUDAPVSGAIN*) pp;
  cudaFree(p->deviceFrame);
  return OK;
} 

static void handleCudaError (CSOUND *csound, cudaError_t error) {
  if (error!= cudaSuccess) {
    csound->Message(csound, "%s in %s at line %d\n", cudaGetErrorString(error),__FILE__,
       __LINE__);
    exit(EXIT_FAILURE);
  }
} 

static int cudapvsgainset(CSOUND *csound, CUDAPVSGAIN *p){

    int32 N = p->fa->N;
    int size = (N+2) * sizeof(float);
    int maxBlockDim;
    int SMcount;
    int totNumThreads = (N+2)/2;
    cudaError_t error;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,0);
    maxBlockDim = deviceProp.maxThreadsPerBlock;
    SMcount = deviceProp.multiProcessorCount;
    csound->Message(csound, "cudapvsgain running on device %s (capability %d.%d)\n", deviceProp.name,
       deviceProp.major, deviceProp.minor);
   
    error = cudaMalloc(&p->deviceFrame, size);
    handleCudaError(csound, error);
    cudaMemcpy(p->deviceFrame,p->fa->frame.auxp,size,cudaMemcpyHostToDevice);
    
    p->fout->sliding = 0;
    
    if (p->fout->frame.auxp == NULL ||
          p->fout->frame.size < sizeof(float) * (N + 2))
        csound->AuxAlloc(csound, (N + 2) * sizeof(float), &p->fout->frame);
    
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
    if (UNLIKELY(!(p->fout->format == PVS_AMP_FREQ) ||
                 (p->fout->format == PVS_AMP_PHASE)))
      return csound->InitError(csound, Str("cudapvsgain: signal format "
                                           "must be amp-phase or amp-freq."));
    csound->RegisterDeinitCallback(csound, p, free_device);
    return OK;
}

static int cudapvsgain(CSOUND *csound, CUDAPVSGAIN *p)
{
    int32   framelength = p->fa->N + 2;
    int size = (int) framelength * sizeof(float);
    MYFLT gain = *p->kgain;

    if (p->lastframe < p->fa->framecount) {
      cudaMemcpy(p->deviceFrame, p->fa->frame.auxp, size, cudaMemcpyHostToDevice); 
      devicepvsgain<<<p->gridSize,p->blockSize>>>(p->deviceFrame, gain, framelength);   // KERNEL LAUNCH
      cudaMemcpy(p->fout->frame.auxp, p->deviceFrame, size, cudaMemcpyDeviceToHost);
      p->fout->framecount = p->fa->framecount;
      p->lastframe = p->fout->framecount;
    }

    return OK;
}


static OENTRY localops[] = {
  {"cudapvsgain", sizeof(CUDAPVSGAIN), 0, 3, "f", "fk",
                               (SUBR) cudapvsgainset, (SUBR) cudapvsgain, NULL}
};

extern "C" {
  LINKAGE
}
