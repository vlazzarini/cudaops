<CsoundSynthesizer>
<CsOptions>
--opcode-lib=./libcudapvshift.so
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 128
0dbfs = 1

gifftsize = $FFT
gihopsize = $HOP

instr 1
  kgain linseg .4, p3, 2
  kshift linseg -2000, p3, 2000 
  klowest linseg 20, p3, 400
  asig soundin "syrinx.wav"
  fsig = pvsanal(asig, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  fsigShifted pvshift fsig, kshift, klowest, 1, kgain
  asig pvsynth fsigShifted    
  out asig
endin

instr 2
  kgain linseg .4, p3, 2
  kshift linseg -2000, p3, 2000 
  klowest linseg 20, p3, 400
  asig soundin "syrinx.wav"
  fsig = pvsanal(asig, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  fsigShifted cudapvshift fsig, kshift, klowest, 1, kgain
  asig pvsynth fsigShifted    
  out asig
endin

instr 3
  kgain linseg .4, p3, 2
  kshift linseg -2000, p3, 2000 
  klowest linseg 20, p3, 400
  asig soundin "syrinx.wav"
  fsig = cudanal(asig, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  fsigShifted cudapvshift fsig, kshift, klowest, 1, kgain
  asig cudasynth fsigShifted    
  out asig  
endin

</CsInstruments>
<CsScore>
i $INSTR 0 60
</CsScore>
</CsoundSynthesizer>
