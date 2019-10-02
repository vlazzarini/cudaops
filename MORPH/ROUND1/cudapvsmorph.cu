/* cudapvsmorph */

#include <csdl.h>
#include <pstream.h>

typedef struct _cudapvsmorph {
  OPDS h;
  PVSDAT  *fout;
  PVSDAT  *fin;
  PVSDAT  *ffr;
  MYFLT   *kampDepth;
  MYFLT   *kfreqDepth;
  float*  devOutput;   // pointer to device memory (output frame)
  float*  devInput1;   // pointer to device memory (input frame #1)
  float*  devInput2;   // pointer to device memory (input frame #2)
  int gridSize;        // number of blocks in the grid (1D)
  int blockSize;       // number of threads in one block (1D)
  uint32  lastframe;
} CUDAPVSMORPH;

__global__ void morph(float* output, float* input1, float* input2, float ampCoeff, float freqCoeff, int length) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = i<<1;
  if (j  < length) {
    output[j] = input1[j]*(1.0-ampCoeff) + input2[j]*(ampCoeff);
    output[j+1] = input1[j+1]*(1.0-freqCoeff) + input2[j+1]*(freqCoeff);
  }
}

static int free_device(CSOUND* csound, void* pp){
  CUDAPVSMORPH* p = (CUDAPVSMORPH*) pp;
  cudaFree(p->devOutput);
  cudaFree(p->devInput1);
  cudaFree(p->devInput2);
  return OK;
} 

static void handleCudaError (CSOUND *csound, cudaError_t error) {
  if (error != cudaSuccess) {
    csound->Message(csound, "%s in %s at line %d\n", cudaGetErrorString(error),__FILE__,
       __LINE__);
    exit(EXIT_FAILURE);
  }
} 

static int cudapvsmorphset(CSOUND *csound, CUDAPVSMORPH *p)
{
  int32 N = p->fin->N;
  int framelength = N+2;
  int size = framelength * sizeof(float);
  int maxBlockDim;
  int SMcount;
  int totNumThreads = framelength>>1;
  cudaError_t error;

  // get info about device
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp,0);
  maxBlockDim = deviceProp.maxThreadsPerBlock;
  SMcount = deviceProp.multiProcessorCount;
  csound->Message(csound, "cudapvsmorph running on device %s (capability %d.%d)\n", deviceProp.name,
     deviceProp.major, deviceProp.minor);

  // device memory allocation (output frame)
  error = cudaMalloc(&p->devOutput, size);
  handleCudaError(csound, error);
  cudaMemset(p->devOutput,0,size);

  // device memory allocation (input 1)
  error = cudaMalloc(&p->devInput1, size);
  handleCudaError(csound, error);

  // device memory allocation (input 2)
  error = cudaMalloc(&p->devInput2, size);
  handleCudaError(csound, error);

  if (p->fout->frame.auxp==NULL || p->fout->frame.size<(N+2)*sizeof(float))
    csound->AuxAlloc(csound,(N+2)*sizeof(float),&p->fout->frame);

  p->blockSize = (((totNumThreads/SMcount)/32)+1)*32;
  if (p->blockSize > maxBlockDim) p->blockSize = maxBlockDim;
  p->gridSize = totNumThreads / p->blockSize + 1;
  p->fout->N =  N;
  p->fout->overlap = p->fin->overlap;
  p->fout->winsize = p->fin->winsize;
  p->fout->wintype = p->fin->wintype;
  p->fout->format = p->fin->format;
  p->fout->framecount = 1;
  p->lastframe = 0;

  if (UNLIKELY(!(p->fout->format==PVS_AMP_FREQ) || (p->fout->format==PVS_AMP_PHASE))) {
    return csound->InitError(csound, Str("signal format must be amp-phase ""or amp-freq.""\n"));
  }

  csound->RegisterDeinitCallback(csound, p, free_device);
    
  return OK;
}

static int cudapvsmorph(CSOUND *csound, CUDAPVSMORPH *p)
{
  int32 N = p->fout->N;
  int framelength = N + 2;
  int size = framelength * sizeof(float);
  float ampDepth = (float) *p->kampDepth;
  float freqDepth = (float) *p->kfreqDepth;
  float *fi1 = (float *) p->fin->frame.auxp;
  float *fi2 = (float *) p->ffr->frame.auxp;
  float *fout = (float *) p->fout->frame.auxp;

  if (UNLIKELY(fout==NULL)) goto err1;

  if (p->lastframe < p->fin->framecount) {
    
    cudaMemcpy(p->devInput1,fi1,size,cudaMemcpyHostToDevice);
    cudaMemcpy(p->devInput2,fi2,size,cudaMemcpyHostToDevice);  

    ampDepth = ampDepth > 0 ? (ampDepth <= 1 ? ampDepth : FL(1.0)): FL(0.0);
    freqDepth = freqDepth > 0 ? (freqDepth <= 1 ? freqDepth : FL(1.0)): FL(0.0);

    morph<<<p->gridSize,p->blockSize>>>(p->devOutput, p->devInput1, p->devInput2, ampDepth, freqDepth, framelength);

    cudaMemcpy(fout,p->devOutput,size,cudaMemcpyDeviceToHost);   
 
    p->fout->framecount = p->lastframe = p->fin->framecount;
  }

  return OK;
  err1:
    return csound->PerfError(csound, p->h.insdshead,
                             Str("cudapvsmorph: not initialised\n"));
}

static OENTRY localops[] = {
 {"cudapvsmorph", sizeof(CUDAPVSMORPH), 0,3,
   "f", "ffkk", (SUBR) cudapvsmorphset, (SUBR) cudapvsmorph}
};

extern "C" {
  LINKAGE
}