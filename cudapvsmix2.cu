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

typedef struct _cudapvsmix2 {
    OPDS    h;
    PVSDAT  *fout;
    PVSDAT  *fa;
    PVSDAT  *fb;
    int     gridSize;     // number of blocks in the grid (1D)
    int     blockSize;    // number of threads in one block (1D)
    uint32  lastframe;
} CUDAPVSMIX2;

__global__ void naivekernel(float* output, float* frameA, float* frameB, int chans) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = i<<1;
  if (i < chans) {
    int test = frameA[j] >= frameB[j];
    if (test) {
      output[j] = frameA[j];
      output[j+1] = frameA[j+1];
    }
    else {
      output[j] = frameB[j];
      output[j+1] = frameB[j+1];
    } 
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
  CUDAPVSMIX2* p = (CUDAPVSMIX2*) pp;
  cudaFree(p->fout->frame.auxp); 
  return OK;
} 


static int cudapvsmix2set(CSOUND *csound, CUDAPVSMIX2 *p)
{
  int32    N = p->fa->N;
  int framelength = N+2;
  int chans = framelength>>1;
  int size = framelength*sizeof(float);

  int maxBlockDim;
  int SMcount;
  int totNumThreads = chans;

  // get info about device
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp,0);
  maxBlockDim = deviceProp.maxThreadsPerBlock;
  SMcount = deviceProp.multiProcessorCount;
  csound->Message(csound, "cudapvsmix2 running on device %s (capability %d.%d)\n", deviceProp.name,
     deviceProp.major, deviceProp.minor);

  /* if (UNLIKELY(p->fa == p->fout || p->fb == p->fout))
     csound->Warning(csound, Str("Unsafe to have same fsig as in and out"));*/
  
  p->fout->sliding = 0;

  if (p->fout->frame.auxp == NULL || p->fout->frame.size < sizeof(float) * (N + 2))
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
 
  csound->RegisterDeinitCallback(csound, p, free_device);
  
  return OK;
}

static int cudapvsmix2(CSOUND *csound, CUDAPVSMIX2 *p)
{
 
  int32    N = p->fa->N;
  int framelength = N+2;
  int chans = framelength>>1;
  float* fout = (float *) p->fout->frame.auxp;
  float* fa = (float *) p->fa->frame.auxp;
  float* fb = (float *) p->fb->frame.auxp;

  if (UNLIKELY(!fsigs_equal(p->fa, p->fb))) goto err1;

  if (p->lastframe < p->fa->framecount) {

    naivekernel<<<p->gridSize,p->blockSize>>>(fout, fa, fb, chans); 

    p->fout->framecount =  p->fa->framecount;
    p->lastframe = p->fout->framecount;
  }

  return OK;
 err1:
    return csound->PerfError(csound, p->h.insdshead,
                             Str("cudapvsmix2: formats are different."));
}

static OENTRY localops[] = {
  {"cudapvsmix2", sizeof(CUDAPVSMIX2),0, 3, "f", "ff", (SUBR) cudapvsmix2set, (SUBR)cudapvsmix2, NULL}
};

extern "C" {
  LINKAGE
}