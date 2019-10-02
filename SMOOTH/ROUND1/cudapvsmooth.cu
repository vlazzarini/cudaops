/* cudapvsmooth */

#include <csdl.h>
#include <pstream.h>

typedef struct _cudapvsmooth {
  OPDS    h;
  PVSDAT  *fout;
  PVSDAT  *fin;
  MYFLT   *kfra;
  MYFLT   *kfrf; 
  AUXCH   del;
  float*  deviceInput;   // pointer to device memory (input frame)
  float*  deviceOutput;   // pointer to device memory (output frame)
  int  gridSize;   // number of blocks in the grid (1D)
  int  blockSize;   // number of threads in one block (1D)
  uint32  lastframe;
} CUDAPVSMOOTH;

// kernel for smoothing the time evolution functions of each channel (both amp and freq)
__global__ void smoothing(float* input, float* output, double alpha, double beta, int length) {
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = i<<1;
  if (j < length) {
    output[j] = (float) (input[j] * (1.0 + alpha) - output[j] * alpha);
    output[j+1] = (float) (input[j+1] * (1.0 + beta) - output[j+1] * beta);
  }
} 

static int free_device(CSOUND* csound, void* pp){
  CUDAPVSMOOTH* p = (CUDAPVSMOOTH*) pp;
  cudaFree(p->deviceInput);
  cudaFree(p->deviceOutput);
  return OK;
} 

static void handleCudaError (CSOUND *csound, cudaError_t error) {
  if (error!= cudaSuccess) {
    csound->Message(csound, "%s in %s at line %d\n", cudaGetErrorString(error),__FILE__,
       __LINE__);
    exit(EXIT_FAILURE);
  }
} 

static int cudapvsmoothset(CSOUND *csound, CUDAPVSMOOTH *p)
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
  csound->Message(csound, "cudapvsmooth running on device %s (capability %d.%d)\n", deviceProp.name,
     deviceProp.major, deviceProp.minor);
  
  // device memory allocation (input data) 
  error = cudaMalloc(&p->deviceInput, size);
  handleCudaError(csound, error);

  // device memory allocation and set to zero (output data to be computed)
  error = cudaMalloc(&p->deviceOutput, size);
  handleCudaError(csound, error);
  cudaMemset(p->deviceOutput,0,size);

  if (UNLIKELY(p->fin == p->fout))
      csound->Warning(csound, Str("Unsafe to have same fsig as in and out"));
  p->fout->NB = (N/2)+1;
  p->fout->sliding = 0;
  if (p->fout->frame.auxp == NULL || p->fout->frame.size < sizeof(float) * (N + 2))
      csound->AuxAlloc(csound, (N + 2) * sizeof(float), &p->fout->frame);
  
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
  
  if (UNLIKELY(!(p->fout->format == PVS_AMP_FREQ) || (p->fout->format == PVS_AMP_PHASE)))
    return csound->InitError(csound, Str("cudapvsmooth: signal format must be amp-phase or amp-freq."));

  csound->RegisterDeinitCallback(csound, p, free_device);

  return OK;
}

static int cudapvsmooth(CSOUND *csound, CUDAPVSMOOTH *p) {
  int N = p->fout->N;
  int framelength = N+2;
  int size = framelength * sizeof(float);
  double ffa = (double) *p->kfra;  // cutoff frequency in fractions of PI (for the amplitude stream)
  double ffr = (double) *p->kfrf;  // cutoff frequency in fractions of PI (for the frequency stream)

  if (p->lastframe < p->fin->framecount) {
    float   *fout, *fin;
    double  costh1, costh2, coef1, coef2;
    fout = (float *) p->fout->frame.auxp;
    fin = (float *) p->fin->frame.auxp;

    cudaMemcpy(p->deviceInput,fin,size,cudaMemcpyHostToDevice);

    ffa = ffa < FL(0.0) ? FL(0.0) : (ffa > FL(1.0) ? FL(1.0) : ffa);
    ffr = ffr < FL(0.0) ? FL(0.0) : (ffr > FL(1.0) ? FL(1.0) : ffr);
    costh1 = 2.0 - cos(PI * ffa);
    costh2 = 2.0 - cos(PI * ffr);
    coef1 = sqrt(costh1 * costh1 - 1.0) - costh1;
    coef2 = sqrt(costh2 * costh2 - 1.0) - costh2;
    
    // channel by channel parallel filtering on the GPU (both amp and freq):
    smoothing<<<p->gridSize,p->blockSize>>>(p->deviceInput, p->deviceOutput, coef1, coef2, framelength);
    
    cudaMemcpy(fout,p->deviceOutput,size,cudaMemcpyDeviceToHost);

    p->fout->framecount = p->lastframe = p->fin->framecount; 
  } 
 
  return OK;
 }


static OENTRY localops[] = {
  {"cudapvsmooth", sizeof(CUDAPVSMOOTH),0, 3, "f", "fxx", (SUBR) cudapvsmoothset,
   (SUBR) cudapvsmooth, NULL}
};

extern "C" {
  LINKAGE
}

