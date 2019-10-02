/* cudapvsmix */

#include <csdl.h>
#include <pstream.h>

typedef struct _cudapvsmix {
    OPDS    h;
    PVSDAT  *fout;
    PVSDAT  *fa;
    PVSDAT  *fb;
    float*  devOutput;    // device memory pointer (output PV frame)
    float*  devFrameA;    // device memory pointer (PV frame #1)
    float*  devFrameB;    // device memory pointer (PV frame #2)  
    int     gridSize;     // number of blocks in the grid (1D)
    int     blockSize;    // number of threads in one block (1D)
    uint32  lastframe;
} CUDAPVSMIX;

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
  CUDAPVSMIX* p = (CUDAPVSMIX*) pp;
  cudaFree(p->devOutput); 
  cudaFree(p->devFrameA);
  cudaFree(p->devFrameB);
  return OK;
} 

static void handleCudaError (CSOUND *csound, cudaError_t error) {
  if (error != cudaSuccess) {
    csound->Message(csound, "%s in %s at line %d\n", cudaGetErrorString(error),__FILE__,
       __LINE__);
    exit(EXIT_FAILURE);
  }
} 

static int cudapvsmixset(CSOUND *csound, CUDAPVSMIX *p)
{
  int32    N = p->fa->N;
  int framelength = N+2;
  int chans = framelength>>1;
  int size = framelength*sizeof(float);

  int maxBlockDim;
  int SMcount;
  int totNumThreads = chans;
  cudaError_t error;

  // get info about device
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp,0);
  maxBlockDim = deviceProp.maxThreadsPerBlock;
  SMcount = deviceProp.multiProcessorCount;
  csound->Message(csound, "cudapvsmix running on device %s (capability %d.%d)\n", deviceProp.name,
     deviceProp.major, deviceProp.minor);

  // device memory allocation (PV frame A)
  error = cudaMalloc(&p->devFrameA, size);
  handleCudaError(csound, error);

  // device memory allocation (PV frame B)
  error = cudaMalloc(&p->devFrameB, size);
  handleCudaError(csound, error);

  // device memory allocation (PV output frame)
  error = cudaMalloc(&p->devOutput, size);
  handleCudaError(csound, error);
  cudaMemset(p->devOutput,0,size);

  /* if (UNLIKELY(p->fa == p->fout || p->fb == p->fout))
     csound->Warning(csound, Str("Unsafe to have same fsig as in and out"));*/
  p->fout->sliding = 0;

  if (p->fout->frame.auxp == NULL || p->fout->frame.size < sizeof(float) * (N + 2))
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
  if (UNLIKELY(!(p->fout->format == PVS_AMP_FREQ) || (p->fout->format == PVS_AMP_PHASE)))
    return csound->InitError(csound, Str("cudapvsmix: signal format "
                                           "must be amp-phase or amp-freq."));
 
  csound->RegisterDeinitCallback(csound, p, free_device);
  
  return OK;
}

static int cudapvsmix(CSOUND *csound, CUDAPVSMIX *p)
{
 
  int32    N = p->fa->N;
  int framelength = N+2;
  int chans = framelength>>1;
  int size = framelength*sizeof(float);
  float* fout = (float *) p->fout->frame.auxp;
  float* fa = (float *) p->fa->frame.auxp;
  float* fb = (float *) p->fb->frame.auxp;

  if (UNLIKELY(!fsigs_equal(p->fa, p->fb))) goto err1;

  if (p->lastframe < p->fa->framecount) {
  
    cudaMemcpy(p->devFrameA, fa, size, cudaMemcpyHostToDevice);
    cudaMemcpy(p->devFrameB, fb, size, cudaMemcpyHostToDevice);

    naivekernel<<<p->gridSize,p->blockSize>>>(p->devOutput, p->devFrameA, p->devFrameB, chans); 
    
    cudaMemcpy(fout, p->devOutput, size, cudaMemcpyDeviceToHost); 

    p->fout->framecount =  p->fa->framecount;
    p->lastframe = p->fout->framecount;
  }

  return OK;
 err1:
    return csound->PerfError(csound, p->h.insdshead,
                             Str("cudapvsmix: formats are different."));
}

static OENTRY localops[] = {
  {"cudapvsmix", sizeof(CUDAPVSMIX),0, 3, "f", "ff", (SUBR) cudapvsmixset, (SUBR)cudapvsmix, NULL}
};

extern "C" {
  LINKAGE
}