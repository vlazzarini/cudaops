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

// to enhance performance, it might be a good idea not to use cudaMemcpy
// with DeviceToDevice specifier: instead, write a kernel just to copy and
// paste data from one location to the other. (it seems that cudaMemcpy is 
// quite slow even when transferring data internally)

#include <csdl.h>
#include <pstream.h>

#define SR (csound->GetSr(csound))

static void AuxCudaAlloc(int size, AUXCH *p){
  float *mem;
  cudaMalloc(&mem, size);
  cudaMemset(mem, 0, size);  
  p->auxp = mem;
  p->size = size;
}

typedef struct _cudapvsblur2{
  OPDS h;
  PVSDAT* fout;
  PVSDAT* fin;
  MYFLT* kdel;         // averaging window length (in sec)
  MYFLT* maxdel;       // maximum expected time for the averaging window (in sec)  
  float* deviceMatrix; // pointer to decive memory (matrix to store current and 
                       // past frames)
  int gridSize;        // number of blocks in the grid (1D)
  int blockSize;       // number of threads in one block (1D)
  MYFLT frpsec;        // frames per second
  int32 count;
  uint32 lastframe; 
} CUDAPVSBLUR2;

__global__ void initialize(float* matrix, float sr, int numFrames, int length) {
  int frame = blockIdx.y*blockDim.y + threadIdx.y;
  int chan = (blockIdx.x*blockDim.x + threadIdx.x) << 1;
  if ((frame < numFrames) && (chan < length)) {
    matrix[frame*length+chan] = 0.0f;
    matrix[frame*length+chan+1] = chan * sr / (length-2);
  } 
}

// naive approach
__global__ void blurnaive(float* matrix, float* output, int firstFrame, int numFrames, int frameCount, int max, int length){
  // int frame = firstFrame + (blockIdx.y*blockDim.y+ threadIdx.y);
  int chan = (blockIdx.x*blockDim.x+ threadIdx.x)<<1;
  float amp = 0.0f;
  float freq = 0.0f;
  int frame;
  if (chan < length) {
    for (frame = firstFrame; frame != frameCount; frame = (frame + 1) % max) { 
      amp += matrix[frame*length+chan]; 
      freq += matrix[frame*length+chan+1];
    }
    output[chan] = (float) (amp / numFrames);
    output[chan+1] = (float) (freq / numFrames);
  }
}

static int free_device(CSOUND* csound, void* pp){
  CUDAPVSBLUR2* p = (CUDAPVSBLUR2*) pp;
  cudaFree(p->fout->frame.auxp);
  cudaFree(p->deviceMatrix);
  return OK;
} 


static void handleCudaError (CSOUND *csound, cudaError_t error) {
  if (error != cudaSuccess) {
    csound->Message(csound, "%s in %s at line %d\n", cudaGetErrorString(error),__FILE__,
       __LINE__);
    exit(EXIT_FAILURE);
  }
} 

static int cudapvsblur2set(CSOUND *csound, CUDAPVSBLUR2 *p)
{
  int32   N = p->fin->N;
  int     olap = p->fin->overlap;   // this is hopsize actually?
  int     maxframes, framelength = N + 2;
  int size = (N+2) * sizeof(float);
  int bigSize;
  int maxBlockDim;
  int SMcount;
  int totNumThreads = (N+2)>>1;  
  cudaError_t error;

  // get info about device
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp,0);
  maxBlockDim = deviceProp.maxThreadsPerBlock;
  SMcount = deviceProp.multiProcessorCount;
  csound->Message(csound, "cudapvsblur2 running on device %s (capability %d.%d)\n", deviceProp.name,
     deviceProp.major, deviceProp.minor);

  // device memory allocation 
  //error = cudaMalloc(&p->deviceOutput, size);
  //handleCudaError(csound, error);
  //cudaMemset(p->deviceOutput,0,size);

  if (UNLIKELY(p->fin == p->fout))
    csound->Warning(csound, Str("Unsafe to have same fsig as in and out"));
  
  p->frpsec = SR / olap;

  maxframes = (int) (*p->maxdel * p->frpsec);  // max number of frames considered, i.e. number of rows  
                                                      // NEED TO ADD 0.5 BEFORE CASTING TO INT? WHY NOT?
  bigSize = size * maxframes;
    
  // device memory allocation 
  error = cudaMalloc(&p->deviceMatrix, bigSize);
  handleCudaError(csound, error);

  // REVISE THIS...(this might prevent memory coalescence)
  // temporary gridSize and blockSize, just for "initialize" kernel
  dim3 initBlockSize((int) maxBlockDim / maxframes, maxframes, 1);
  dim3 initGridSize((framelength*maxframes>>1) / maxBlockDim + 1, 1, 1);

  // REVISE THIS...(this should be a little better, NOT SURE)
  // NOTE: IF THESE SIZES ARE USED, THE "INITIALIZE" KERNEL NEEDS TO BE CHANGED!
  // temporary gridSize and blockSize, just for "initialize" kernel
  // dim3 initBlockSize(maxBlockDim, 1, 1);
  // dim3 initGridSize((framelength*maxframes>>1) / maxBlockDim + 1, 1, 1);

  initialize<<<initGridSize,initBlockSize>>>(p->deviceMatrix, SR, maxframes, framelength);     

  if (p->fout->frame.auxp == NULL || p->fout->frame.size < sizeof(float) * (N + 2))
    AuxCudaAlloc(size, &p->fout->frame); 

  p->blockSize = (((totNumThreads/SMcount)/32)+1)*32;
  if (p->blockSize > maxBlockDim) p->blockSize = maxBlockDim;
  p->gridSize = totNumThreads / p->blockSize + 1;
  p->fout->N = N;
  p->fout->overlap = olap;
  p->fout->winsize = p->fin->winsize;
  p->fout->wintype = p->fin->wintype;
  p->fout->format = p->fin->format;
  p->fout->framecount = 1;
  p->lastframe = 0;
  p->count = 0;
  p->fout->sliding = 0;
  p->fout->NB = p->fin->NB;

 csound->RegisterDeinitCallback(csound, p, free_device);

  return OK;
}

static int cudapvsblur2(CSOUND *csound, CUDAPVSBLUR2 *p)
{
  int32    N = p->fout->N, first, framelength = N + 2;
  int32    countr = p->count;
  int size = (N+2) * sizeof(float);
  int     delayframes = (int) (*p->kdel * p->frpsec);
  int     maxframes = (int) (*p->maxdel * p->frpsec);
  float   *fin = (float *) p->fin->frame.auxp;
  float   *fout = (float *) p->fout->frame.auxp;

  if (UNLIKELY(fout == NULL)) goto err1;

  if (p->lastframe < p->fin->framecount) {

    delayframes = delayframes >= 0 ? (delayframes < maxframes ? delayframes : maxframes - 1) : 0;

    cudaMemcpy(p->deviceMatrix+(countr*framelength),fin,size,cudaMemcpyDeviceToDevice);
  
    if (delayframes) {
      if ((first = countr - delayframes) < 0)
        first += maxframes;
      
      blurnaive<<<p->gridSize,p->blockSize>>>(p->deviceMatrix, fout, first, delayframes, countr, maxframes, framelength);
 
    }
    else {
      cudaMemcpy(fout, fin, size, cudaMemcpyDeviceToDevice);   // bypass blurring   
    }

    p->fout->framecount = p->lastframe = p->fin->framecount;
    countr++;
    p->count = countr < maxframes ? countr : 0;
  }

  return OK;
  err1: return csound->PerfError(csound, p->h.insdshead,
                             Str("cudapvsblur2: not initialised"));
}

static OENTRY localops[] = {
  {"cudapvsblur2", sizeof(CUDAPVSBLUR2),0, 3, "f", "fki", (SUBR) cudapvsblur2set, (SUBR) cudapvsblur2, NULL}
};

extern "C" {
  LINKAGE
}


