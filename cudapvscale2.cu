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
// Consider writing another "freqScaleBasic" kernel for scaling upwards only
// (without atomic operations, as they are not needed)

#include <csdl.h>
#include <pstream.h>
#include <cufft.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>

static void AuxCudaAlloc(int size, AUXCH *p){
  float *mem;
  cudaMalloc(&mem, size);
  cudaMemset(mem, 0, size);  
  p->auxp = mem;
  p->size = size;
}

typedef struct _cudapvscale2 {
  OPDS    h;
  PVSDAT  *fout;
  PVSDAT  *fin;
  MYFLT   *kscal;
  MYFLT   *keepform;
  MYFLT   *gain;
  MYFLT   *coefs;  
  AUXCH   fenv, ceps;
  float*  deviceEnv;   // pointer to device memory (amplitude spectral envelope frame)
  float*  deviceCepstrum;   // pointer to device memory (cepstrum frame)
  float*  deviceTrueEnv;   // pointer to device memory  (true envelope)
  float*  deviceSmoothTrueEnv;   // pointer to device memory (true envelope, smoothed) 
  int*    deviceMask;   // pointer to device memory (boolean mask for condition checking)   
  int  gridSize;   // number of blocks in the grid (1D)
  int blockSize;   // number of threads in one block (1D)
  cufftHandle forwardPlan;   // forward cuFFT plan
  cufftHandle inversePlan;   // inverse cuFFT plan
  uint32  lastframe;
} CUDAPVSCALE2;

// kernel for frequency scaling without formant keeping (part one)
__global__ void freqScaleBasic(float* input, float* output, MYFLT scaleFactor, int nhalf) {
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = (i<<1) + 2;
  int N = nhalf<<1;
  int newchan;  
  if (i < nhalf-1) {
    newchan = (int)(((i+1)*scaleFactor)+0.5) << 1;
    if (newchan < N && newchan > 0) {
      atomicExch(&output[newchan],input[j]);   // move amplitude data to new positions
      atomicExch(&output[newchan+1], (float)(input[j+1]*scaleFactor));   // change bin frequencies
    }      
  }
}

// kernel for frequency scaling (part two, with or without formant keeping)
__global__ void fixPVandGain(float* input, float* output, float gain, int length) {
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = i<<1;
  if (j < length) {
    if (isnan(output[j]))   // LIKELY, THERE IS A PERFORMANCE LOSS HERE
      output[j] = 0.0f;  // set to zero any invalid amplitude 
    if (output[j+1] == -1.0f) {   // LIKELY, THERE IS A PERFORMANCE LOSS HERE
      output[j] = 0.0f;   // set to zero the amp related to any undefined frequency
    }
    else
      output[j] *= gain;   // scale all amplitudes by the gain factor      
  }
  if (j == 0) output[0] = input[0];   // keep original DC amplitude
  if (j == length-2) output[length-2] = input[length-2];   // keep original Nyquist amplitude
}

__global__ void takeLog(float* input, float* env, int nhalf) {
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = i<<1; 
  if (i < nhalf) {
    env[i] = log(input[j] > 0.0 ? input[j] : 1e-20);   // take the log of the amplitudes
  }
}

__global__ void lifter(float* cepstrum, int nCoefs, int nhalf) {
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int k = i + nCoefs; 
  if (k < nhalf+2-nCoefs) {   
    cepstrum[k] = 0.0;   // kill all the cepstrum coefficients above nCoefs
  }
}

__global__ void expon(float* env, int nhalf) {
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  if (i < nhalf) {
    env[i] = exp(env[i]/nhalf);   // exponentiate
  }
}

// kernel for frequency scaling with formant keeping (part one)
__global__ void freqScaleFormant(float* input, float* output, float* env, MYFLT scaleFactor, float maxAmp, int nhalf) {
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = (i<<1) + 2;
  int N = nhalf<<1;
  int newchan;  
  if (i < nhalf-1) {
    env[i+1] /= maxAmp;   // normalize the spectral envelope
    input[j] /= env[i+1];   // equalize the original amplitudes so that formant shaping is more effective 
    newchan = (int)(((i+1)*scaleFactor)+0.5) << 1;
    if (newchan < N && newchan > 0) { 
      atomicExch(&output[newchan], input[j]*env[newchan>>1]*0.9);   // move amp data to new positions and weight by normalized env
      atomicExch(&output[newchan+1], (float)(input[j+1]*scaleFactor));   // change bin frequencies
    }      
  }
}

// after completing the inverse fft, this kernel updates the true envelope: 
// for each bin, the max of input and smoothed spectral envelopes is taken 
__global__ void update(float* original, float* newTE, float* current, int nhalf) {
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  if (i < nhalf) {
    current[i] /= nhalf; 
    newTE[i] = (original[i] < current[i]) ? current[i] : original[i];   // LIKELY, THERE IS A PERFORMANCE LOSS HERE
  }
}

// kernel for testing the true envelope condition
__global__ void test(float* nonSmoothed, float* smoothed, int* mask, int nhalf) {
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int diff;
  if (i < nhalf) {
    diff = fabs(nonSmoothed[i] - smoothed[i]/nhalf);
    mask[i] = (diff > 0.23) ? 1 : 0;   // WHAT THRESHOLD TO USE?? different behaviour as opposed to CPU version!
  }
}
       
static int free_device(CSOUND* csound, void* pp){
  CUDAPVSCALE2* p = (CUDAPVSCALE2*) pp;
  cudaFree(p->fout->frame.auxp);
  cudaFree(p->deviceEnv);
  cudaFree(p->deviceCepstrum);
  cudaFree(p->deviceTrueEnv);
  cudaFree(p->deviceSmoothTrueEnv);
  cudaFree(p->deviceMask);
  cufftDestroy(p->forwardPlan);
  cufftDestroy(p->inversePlan);
  return OK;
} 

static void handleCudaError (CSOUND *csound, cudaError_t error) {
  if (error!= cudaSuccess) {
    csound->Message(csound, "%s in %s at line %d\n", cudaGetErrorString(error),__FILE__,
       __LINE__);
    exit(EXIT_FAILURE);
  }
} 


static int cudapvscale2set(CSOUND *csound, CUDAPVSCALE2 *p)
{
  int N = p->fin->N;
  int Nhalf = N>>1;
  int size = (N+2) * sizeof(float);
  int smallSize = ((Nhalf>>1)+1) * sizeof(cufftComplex);
  int maxBlockDim;
  int SMcount;
  int totNumThreads = Nhalf;   // TO BE MODIFIED, MAYBE NOT
  cudaError_t error;

  // get info about device
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp,0);
  maxBlockDim = deviceProp.maxThreadsPerBlock;
  SMcount = deviceProp.multiProcessorCount;
  csound->Message(csound, "cudapvscale2 running on device %s (capability %d.%d)\n", deviceProp.name,
     deviceProp.major, deviceProp.minor);
  
  // create a cuFFT plan to use later
  cufftPlan1d(&p->forwardPlan, Nhalf, CUFFT_R2C, 1);   
  cufftPlan1d(&p->inversePlan, Nhalf, CUFFT_C2R, 1);  
  cufftSetCompatibilityMode(p->forwardPlan, CUFFT_COMPATIBILITY_NATIVE);
  cufftSetCompatibilityMode(p->inversePlan, CUFFT_COMPATIBILITY_NATIVE);

  // device memory allocation and set to zero (amplitude spectral envelope, approx half the length)
  error = cudaMalloc(&p->deviceEnv, smallSize);
  handleCudaError(csound, error);
  cudaMemset(p->deviceEnv,0,smallSize);

  // device memory allocation and set to zero (cepstrum, approx half the length)
  error = cudaMalloc(&p->deviceCepstrum, smallSize);
  handleCudaError(csound, error);
  cudaMemset(p->deviceCepstrum,0,smallSize);

  // device memory allocation and set to zero (true envelope, approx half the length)
  error = cudaMalloc(&p->deviceTrueEnv, smallSize);
  handleCudaError(csound, error);
  cudaMemset(p->deviceTrueEnv,0,smallSize);

  // device memory allocation and set to zero (smoothed true envelope, approx half the length)
  error = cudaMalloc(&p->deviceSmoothTrueEnv, smallSize);
  handleCudaError(csound, error);
  cudaMemset(p->deviceSmoothTrueEnv,0,smallSize);

  // device memory allocation and set to one (mask)
  error = cudaMalloc(&p->deviceMask, Nhalf*sizeof(int));
  handleCudaError(csound, error);
  cudaMemset(p->deviceMask,1,Nhalf*sizeof(int));

  if (UNLIKELY(p->fin == p->fout))
    csound->Warning(csound, Str("Unsafe to have same fsig as in and out"));
  p->fout->NB = p->fin->NB;
  
  p->fout->sliding = 0;
 
  if (p->fout->frame.auxp == NULL || 
      p->fout->frame.size < sizeof(float) * (N + 2))  /* RWD MUST be 32bit */
    AuxCudaAlloc(size, &p->fout->frame);
  
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
  
  csound->RegisterDeinitCallback(csound, p, free_device);

  return OK;
}


static int cudapvscale2(CSOUND *csound, CUDAPVSCALE2 *p)
{
    int     i, N = p->fout->N;
    int Nhalf = N>>1;
    int framelength = N+2;
    float   max = 0.0f;
    MYFLT   pscal = (MYFLT) *p->kscal;
    int     keepform = (int) *p->keepform;
    float   g = (float) *p->gain;
    float   *fin = (float *) p->fin->frame.auxp;
    float   *fout = (float *) p->fout->frame.auxp;  
    int coefs = (int) *p->coefs;

    cufftComplex* cufftEnv;
    cufftComplex* cufftCepstrum;
    cufftComplex* cufftTrueEnv;
    cufftComplex* cufftSmoothTrueEnv;

    thrust::device_ptr<float> dev_ptr1 = thrust::device_pointer_cast(fout);
    thrust::device_ptr<float> dev_ptr2 = thrust::device_pointer_cast(p->deviceEnv);
    thrust::device_ptr<float> dev_ptr3 = thrust::device_pointer_cast(p->deviceSmoothTrueEnv);
    thrust::device_ptr<int> dev_ptr4 = thrust::device_pointer_cast(p->deviceMask);

    if (UNLIKELY(fout == NULL)) goto err1;

    if (p->lastframe < p->fin->framecount) {
      
      if (keepform == 0) {
        thrust::fill(dev_ptr1, dev_ptr1+framelength, -1.0f);   // resets the output	
        freqScaleBasic<<<p->gridSize,p->blockSize>>>(fin, fout, pscal, Nhalf);   // freq scaling
        fixPVandGain<<<p->gridSize,p->blockSize>>>(fin, fout, g, framelength);   // apply gain to all amplitudes 
      }

      else if (keepform==1) {
        if (coefs<1) coefs = 80;
        
        thrust::fill(dev_ptr1, dev_ptr1+framelength, -1.0f);   // resets the output
        takeLog<<<p->gridSize,p->blockSize>>>(fin, p->deviceEnv, Nhalf);

        cufftEnv = (cufftComplex*) p->deviceEnv;
        cufftCepstrum = (cufftComplex*) p->deviceCepstrum;

        // take the fft of the log of the spectral envelope... 
        if(cufftExecR2C(p->forwardPlan,(cufftReal*)cufftEnv,cufftCepstrum)!= CUFFT_SUCCESS)
          csound->Message(csound, "CUDA FFT error\n");
        if (cudaDeviceSynchronize() != cudaSuccess)
          csound->Message(csound,"CUDA error: Failed to synchronize\n");
        
        lifter<<<p->gridSize,p->blockSize>>>(p->deviceCepstrum, coefs, Nhalf);   // liftering stage: keep only low quefrency coefficients
         
        // take the inverse fft of the liftered cepstrum...  
        if(cufftExecC2R(p->inversePlan,cufftCepstrum,(cufftReal*)cufftEnv)!= CUFFT_SUCCESS)
          csound->Message(csound, "CUDA FFT error\n");
        if (cudaDeviceSynchronize() != cudaSuccess) 
          csound->Message(csound,"CUDA error: Failed to synchronize\n");
        
        // scale the result of the inverse transform and exponentiate to go back to true amplitudes...
        expon<<<p->gridSize,p->blockSize>>>(p->deviceEnv, Nhalf);   

        max = *(thrust::max_element(dev_ptr2, dev_ptr2+Nhalf));   // find maximum amp in spectral envelope

        freqScaleFormant<<<p->gridSize,p->blockSize>>>(fin, fout, p->deviceEnv, pscal, max, Nhalf);   // normalize spectral env and freq scale the input
        fixPVandGain<<<p->gridSize,p->blockSize>>>(fin, fout, g, framelength);   // apply gain to all amplitudes 
      }
      
      else if (keepform==2) {
        i = 0;   // DEBUG
        int cond = 1;
        if (coefs<1) coefs = 80;
        cufftEnv = (cufftComplex*) p->deviceEnv;
        cufftCepstrum = (cufftComplex*) p->deviceCepstrum;
        cufftTrueEnv = (cufftComplex*) p->deviceTrueEnv;
        cufftSmoothTrueEnv = (cufftComplex*) p->deviceSmoothTrueEnv;

        thrust::fill(dev_ptr1, dev_ptr1+framelength, -1.0f);   // resets the output
        takeLog<<<p->gridSize,p->blockSize>>>(fin, p->deviceEnv, Nhalf);

        // loop initialization stage: smooth the original log spectral envelope... 
        // take the fft of the log of the spectral envelope...  
        if(cufftExecR2C(p->forwardPlan,(cufftReal*)cufftEnv,cufftCepstrum)!= CUFFT_SUCCESS)
          csound->Message(csound, "CUDA FFT error\n");
        if (cudaDeviceSynchronize() != cudaSuccess)
          csound->Message(csound,"CUDA error: Failed to synchronize\n");
        
        lifter<<<p->gridSize,p->blockSize>>>(p->deviceCepstrum, coefs, Nhalf);  // liftering: keep only low quefrency coefficients 

        // take the inverse fft of the liftered cepstrum...  
        if(cufftExecC2R(p->inversePlan,cufftCepstrum,(cufftReal*)cufftSmoothTrueEnv)!= CUFFT_SUCCESS)
          csound->Message(csound, "CUDA FFT error\n");
        if (cudaDeviceSynchronize() != cudaSuccess) 
          csound->Message(csound,"CUDA error: Failed to synchronize\n");

        while(cond) { 
          // i++;   // DEBUG
          update<<<p->gridSize,p->blockSize>>>(p->deviceEnv, p->deviceTrueEnv, p->deviceSmoothTrueEnv, Nhalf);
          
          // take the fft of the true envelope...  
          if(cufftExecR2C(p->forwardPlan,(cufftReal*)cufftTrueEnv,cufftCepstrum)!= CUFFT_SUCCESS)
            csound->Message(csound, "CUDA FFT error\n");
          if (cudaDeviceSynchronize() != cudaSuccess)
            csound->Message(csound,"CUDA error: Failed to synchronize\n");
        
          lifter<<<p->gridSize,p->blockSize>>>(p->deviceCepstrum, coefs, Nhalf); // liftering: keep only low quefrency coefficients 

          // take the inverse fft of the liftered cepstrum... 
          if(cufftExecC2R(p->inversePlan,cufftCepstrum,(cufftReal*)cufftSmoothTrueEnv)!= CUFFT_SUCCESS)
            csound->Message(csound, "CUDA FFT error\n");
          if (cudaDeviceSynchronize() != cudaSuccess) 
            csound->Message(csound,"CUDA error: Failed to synchronize\n");
      
          test<<<p->gridSize,p->blockSize>>>(p->deviceTrueEnv, p->deviceSmoothTrueEnv, p->deviceMask, Nhalf);
          if((thrust::reduce(dev_ptr4, dev_ptr4+Nhalf)) == 0)
            cond = 0; 
        }
        // printf("%d\n", i);   // DEBUG

        // scale the result of the inverse transform and exponentiate to go back to true amplitudes...
        expon<<<p->gridSize,p->blockSize>>>(p->deviceSmoothTrueEnv, Nhalf);   

        max = *(thrust::max_element(dev_ptr3, dev_ptr3+Nhalf));   // find maximum amp in spectral envelope

        freqScaleFormant<<<p->gridSize,p->blockSize>>>(fin, fout, p->deviceSmoothTrueEnv, pscal, max, Nhalf);   // normalize spectral env and freq scale the input
        fixPVandGain<<<p->gridSize,p->blockSize>>>(fin, fout, g, framelength);   // apply gain to all amplitudes 
      }

      p->fout->framecount = p->lastframe = p->fin->framecount;
    }

    return OK;

 err1:
    return csound->PerfError(csound, p->h.insdshead,
                             Str("cudapvscale2: not initialised"));
}


static OENTRY localops[] = {
  {"cudapvscale2", sizeof(CUDAPVSCALE2),0, 3, "f", "fxOPO", (SUBR) cudapvscale2set,
   (SUBR) cudapvscale2}
};


extern "C" {
  LINKAGE
}
