/* cudapvsmorph2 */

#include <csdl.h>
#include <pstream.h>

static void AuxCudaAlloc(int size, AUXCH *p){
  float *mem;
  cudaMalloc(&mem, size);
  cudaMemset(mem, 0, size);  
  p->auxp = mem;
  p->size = size;
}

typedef struct _cudapvsmorph2 {
  OPDS h;
  PVSDAT  *fout;
  PVSDAT  *fin;
  PVSDAT  *ffr;
  MYFLT   *kampDepth;
  MYFLT   *kfreqDepth;
  int gridSize;        // number of blocks in the grid (1D)
  int blockSize;       // number of threads in one block (1D)
  uint32  lastframe;
} CUDAPVSMORPH2;

__global__ void morph(float* output, float* input1, float* input2, float ampCoeff, float freqCoeff, int length) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = i<<1;
  if (j  < length) {
    output[j] = input1[j]*(1.0-ampCoeff) + input2[j]*(ampCoeff);
    output[j+1] = input1[j+1]*(1.0-freqCoeff) + input2[j+1]*(freqCoeff);
  }
}

static int free_device(CSOUND* csound, void* pp){
  CUDAPVSMORPH2* p = (CUDAPVSMORPH2*) pp;
  cudaFree(p->fout->frame.auxp);
  return OK;
} 


static int cudapvsmorph2set(CSOUND *csound, CUDAPVSMORPH2 *p)
{
  int32 N = p->fin->N;
  int framelength = N+2;
  int size = framelength * sizeof(float);
  int maxBlockDim;
  int SMcount;
  int totNumThreads = framelength>>1;

  // get info about device
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp,0);
  maxBlockDim = deviceProp.maxThreadsPerBlock;
  SMcount = deviceProp.multiProcessorCount;
  csound->Message(csound, "cudapvsmorph2 running on device %s (capability %d.%d)\n", deviceProp.name,
     deviceProp.major, deviceProp.minor);

  if (p->fout->frame.auxp==NULL || p->fout->frame.size<(N+2)*sizeof(float))
    AuxCudaAlloc(size, &p->fout->frame);

  p->blockSize = (((totNumThreads/SMcount)/32)+1)*32;
  if (p->blockSize > maxBlockDim) p->blockSize = maxBlockDim;
  p->gridSize = totNumThreads / p->blockSize + 1;
  p->fout->N =  N;
  p->fout->overlap = p->fin->overlap;
  p->fout->winsize = p->fin->winsize;
  p->fout->wintype = p->fin->wintype;
  p->fout->format = p->fin->format;
  p->fout->sliding = 0;
  p->fout->framecount = 1;
  p->lastframe = 0;

  csound->RegisterDeinitCallback(csound, p, free_device);
    
  return OK;
}

static int cudapvsmorph2(CSOUND *csound, CUDAPVSMORPH2 *p)
{
  int32 N = p->fout->N;
  int framelength = N + 2;
  float ampDepth = (float) *p->kampDepth;
  float freqDepth = (float) *p->kfreqDepth;
  float *fi1 = (float *) p->fin->frame.auxp;
  float *fi2 = (float *) p->ffr->frame.auxp;
  float *fout = (float *) p->fout->frame.auxp;

  if (UNLIKELY(fout==NULL)) goto err1;

  if (p->lastframe < p->fin->framecount) { 

    ampDepth = ampDepth > 0 ? (ampDepth <= 1 ? ampDepth : FL(1.0)): FL(0.0);
    freqDepth = freqDepth > 0 ? (freqDepth <= 1 ? freqDepth : FL(1.0)): FL(0.0);

    morph<<<p->gridSize,p->blockSize>>>(fout, fi1, fi2, ampDepth, freqDepth, framelength);   
 
    p->fout->framecount = p->lastframe = p->fin->framecount;
  }

  return OK;

  err1:
    return csound->PerfError(csound, p->h.insdshead,
                             Str("cudapvsmorph2: not initialised\n"));
}

static OENTRY localops[] = {
 {"cudapvsmorph2", sizeof(CUDAPVSMORPH2), 0,3,
   "f", "ffkk", (SUBR) cudapvsmorph2set, (SUBR) cudapvsmorph2}
};

extern "C" {
  LINKAGE
}