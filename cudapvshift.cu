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
#include <cufft.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>


typedef struct _cudapvshift {
  OPDS    h;
  PVSDAT  *fout;
  PVSDAT  *fin;
  MYFLT   *kshift;
  MYFLT   *lowest;
  MYFLT   *keepform;
  MYFLT   *gain;
  MYFLT   *coefs;  
  AUXCH   fenv, ceps;
  float*  deviceInput;   // pointer to device memory (input frame)
  float*  deviceOutput;   // pointer to device memory (output frame)
  float*  deviceEnv;   // pointer to device memory (amplitude spectral envelope frame)
  float*  deviceCepstrum;   // pointer to device memory (cepstrum frame)
  float*  deviceTrueEnv;   // pointer to device memory  (true envelope)
  float*  deviceSmoothTrueEnv;   // pointer to device memory (true envelope, smoothed) 
  int*    deviceMask;   // pointer to device memory (boolean mask for condition checking)   
  int  gridSize;   // number of blocks in the grid (1D)
  int  blockSize;   // number of threads in one block (1D)
  cufftHandle forwardPlan;   // forward cuFFT plan
  cufftHandle inversePlan;   // inverse cuFFT plan
  uint32  lastframe;
} CUDAPVSHIFT;

// kernel for frequency shifting without formant keeping (part one)
__global__ void freqShiftBasic(float* input, float* output, MYFLT shift, int shiftChan, int lowestIndx, int nhalf) {
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = i<<1;
  int lowestChan = lowestIndx>>1;
  int N = nhalf<<1;
  int newchan;  
  if (i < lowestChan) {
    // leave PV data as it is below a certain channel:
    output[j] = input[j];
    output[j+1] = input[j+1];
  }
  if (i >= lowestChan && i < nhalf) {
    newchan = (i + shiftChan) << 1;
    if (newchan < N && newchan >= lowestIndx) {
      output[newchan] = input[j];   // move amplitude data to new positions
      output[newchan+1] = (float) (input[j+1] + shift);   // change bin frequencies
    }      
  }
}

// kernel for frequency shifting (part two, with or without formant keeping)
__global__ void fixPVandGain(float* output, float gain, int lowestIndx, int length) {
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = i<<1;
  if (j >= lowestIndx && j < length) {
    if (isnan(output[j]))   // LIKELY, THERE IS A PERFORMANCE LOSS HERE
      output[j] = 0.0f;  // set to zero any invalid amplitude 
    if (output[j+1] == -1.0f) {   // LIKELY, THERE IS A PERFORMANCE LOSS HERE
      output[j] = 0.0f;   // set to zero the amp related to any undefined frequency
    }
    else
      output[j] *= gain;   // scale all amplitudes by the gain factor      
  }
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

// kernel for frequency shifting with formant keeping (part one)
__global__ void freqShiftFormant(float* input, float* output, float* env, MYFLT shift, int shiftChan, int lowestIndx, float maxAmp, int nhalf) {
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = i<<1;
  int lowestChan = lowestIndx>>1;
  int N = nhalf<<1;
  int newchan;
  if (i < lowestChan) {
    // leave PV data as it is below a certain channel:
    output[j] = input[j];
    output[j+1] = input[j+1];
  }
  else if (i < nhalf) {
    env[i] /= maxAmp;   // normalize the spectral envelope
    input[j] /= env[i];   // equalize the original amplitudes so that formant shaping is more effective 
    newchan = (i + shiftChan) << 1;
    if (newchan < N && newchan >= lowestIndx) { 
      output[newchan] = input[j]*env[newchan>>1]*0.9;   // move amp data to new positions and weight by normalized env
      output[newchan+1] = (float)(input[j+1] + shift);   // change bin frequencies
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
    mask[i] = (diff > 0.23) ? 1 : 0;   // WHAT THRESHOLD TO USE?? different behaviour when compared to CPU version!
  }
}
       
static int free_device(CSOUND* csound, void* pp){
  CUDAPVSHIFT* p = (CUDAPVSHIFT*) pp;
  cudaFree(p->deviceInput);
  cudaFree(p->deviceOutput);
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


static int cudapvshiftset(CSOUND *csound, CUDAPVSHIFT *p)
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
  csound->Message(csound, "cudapvshift running on device %s (capability %d.%d)\n", deviceProp.name,
     deviceProp.major, deviceProp.minor);
  
  // create a cuFFT plan to use later
  cufftPlan1d(&p->forwardPlan, Nhalf, CUFFT_R2C, 1);   
  cufftPlan1d(&p->inversePlan, Nhalf, CUFFT_C2R, 1);  
  cufftSetCompatibilityMode(p->forwardPlan, CUFFT_COMPATIBILITY_NATIVE);
  cufftSetCompatibilityMode(p->inversePlan, CUFFT_COMPATIBILITY_NATIVE);
  
  // device memory allocation and copy from host (input data) 
  error = cudaMalloc(&p->deviceInput, size);
  handleCudaError(csound, error);
  
  // device memory allocation and set to zero (output data to be computed)
  error = cudaMalloc(&p->deviceOutput, size);
  handleCudaError(csound, error);
  cudaMemset(p->deviceOutput,0,size);

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
  
  if (p->fout->frame.auxp == NULL || p->fout->frame.size < sizeof(float) * (N + 2))  /* RWD MUST be 32bit */
      csound->AuxAlloc(csound, (N + 2) * sizeof(float), &p->fout->frame);
  else memset(p->fout->frame.auxp, 0, (N+2)*sizeof(float));
    
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
  p->fout->sliding = p->fin->sliding;
  p->fout->NB = p->fin->NB;

  if (p->ceps.auxp == NULL || p->ceps.size < sizeof(MYFLT) * (N+2))
    csound->AuxAlloc(csound, sizeof(MYFLT) * (N + 2), &p->ceps);
  else memset(p->ceps.auxp, 0, sizeof(MYFLT)*(N+2));
  if (p->fenv.auxp == NULL || p->fenv.size < sizeof(MYFLT) * (N+2))
    csound->AuxAlloc(csound, sizeof(MYFLT) * (N + 2), &p->fenv);
  else memset(p->fenv.auxp, 0, sizeof(MYFLT)*(N+2));

  csound->RegisterDeinitCallback(csound, p, free_device);

  return OK;
}

static int cudapvshift(CSOUND *csound, CUDAPVSHIFT *p)
{
    int     i, N = p->fout->N;
    int Nhalf = N>>1;
    int framelength = N+2;
    int size = framelength * sizeof(float);
    float   max = 0.0f;
    MYFLT   pshift = (MYFLT) *p->kshift;
    int     cshift = (int) (pshift * N * (1.0/csound->GetSr(csound)));
    int     lowest = abs((int) (*p->lowest * N * (1.0/csound->GetSr(csound))));
    int     keepform = (int) *p->keepform;
    float   g = (float) *p->gain;
    float   *fin = (float *) p->fin->frame.auxp;
    float   *fout = (float *) p->fout->frame.auxp;  
    int coefs = (int) *p->coefs;

    cufftComplex* cufftEnv;
    cufftComplex* cufftCepstrum;
    cufftComplex* cufftTrueEnv;
    cufftComplex* cufftSmoothTrueEnv;

    thrust::device_ptr<float> dev_ptr1 = thrust::device_pointer_cast(p->deviceOutput);
    thrust::device_ptr<float> dev_ptr2 = thrust::device_pointer_cast(p->deviceEnv);
    thrust::device_ptr<float> dev_ptr3 = thrust::device_pointer_cast(p->deviceSmoothTrueEnv);
    thrust::device_ptr<int> dev_ptr4 = thrust::device_pointer_cast(p->deviceMask);

    if (UNLIKELY(fout == NULL)) goto err1;

    if (p->lastframe < p->fin->framecount) {

      lowest = lowest ? (lowest > Nhalf ? Nhalf : lowest<<1) : 2;

      cudaMemcpy(p->deviceInput,fin,size,cudaMemcpyHostToDevice);
      
      if (keepform == 0) {
        thrust::fill(dev_ptr1, dev_ptr1+framelength, -1.0f);   // resets the output	
        freqShiftBasic<<<p->gridSize,p->blockSize>>>(p->deviceInput, p->deviceOutput, pshift, cshift, lowest, Nhalf);   // freq shifting
        fixPVandGain<<<p->gridSize,p->blockSize>>>(p->deviceOutput, g, lowest, framelength);   // apply gain to all amplitudes 
      }

      else if (keepform==1) {
        if (coefs<1) coefs = 80;
        
        thrust::fill(dev_ptr1, dev_ptr1+framelength, -1.0f);   // resets the output
        takeLog<<<p->gridSize,p->blockSize>>>(p->deviceInput, p->deviceEnv, Nhalf);

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

        freqShiftFormant<<<p->gridSize,p->blockSize>>>(p->deviceInput, p->deviceOutput, p->deviceEnv, pshift, cshift, lowest, max, Nhalf);   // normalize spectral env and freq scale the input
        fixPVandGain<<<p->gridSize,p->blockSize>>>(p->deviceOutput, g, lowest, framelength);   // apply gain to all amplitudes 
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
        takeLog<<<p->gridSize,p->blockSize>>>(p->deviceInput, p->deviceEnv, Nhalf);

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
          i++;   // DEBUG
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
        printf("%d\n", i);   // DEBUG

        // scale the result of the inverse transform and exponentiate to go back to true amplitudes...
        expon<<<p->gridSize,p->blockSize>>>(p->deviceSmoothTrueEnv, Nhalf);   

        max = *(thrust::max_element(dev_ptr3, dev_ptr3+Nhalf));   // find maximum amp in spectral envelope

        freqShiftFormant<<<p->gridSize,p->blockSize>>>(p->deviceInput, p->deviceOutput, p->deviceSmoothTrueEnv, pshift, cshift, lowest, max, Nhalf);   // normalize spectral env and freq scale the input
        fixPVandGain<<<p->gridSize,p->blockSize>>>(p->deviceOutput, g, lowest, framelength);   // apply gain to all amplitudes 
      }
      cudaMemcpy(fout, p->deviceOutput, size, cudaMemcpyDeviceToHost);
      fout[0] = fin[0];   // keep original DC amplitude
      fout[N] = fin[N];   // keep original Nyquist amplitude
      p->fout->framecount = p->lastframe = p->fin->framecount;
    }

    return OK;

 err1:
    return csound->PerfError(csound, p->h.insdshead,
                             Str("cudapvshift: not initialised"));
}


static OENTRY localops[] = {
  {"cudapvshift", sizeof(CUDAPVSHIFT),0, 3, "f", "fxkOPO", (SUBR) cudapvshiftset,
   (SUBR) cudapvshift}
};


extern "C" {
  LINKAGE
}

