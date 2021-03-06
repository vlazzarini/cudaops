<CsoundSynthesizer>
<CsOptions>
--opcode-lib=../../libcudapvs2.dylib
--opcode-lib=./libcudapvshift2.so
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 128
0dbfs = 1

gifftsize = $FFT
gihopsize = $HOP

instr 4
  kgain linseg .4, p3, 2
  kshift linseg -2000, p3, 2000 
  klowest linseg 20, p3, 400
  asig soundin "syrinx.wav"
  fsig cudanal2 asig, gifftsize, gihopsize, gifftsize, 1
  fsigShifted cudapvshift2 fsig, kshift, klowest, 1, kgain
  asig cudasynth2 fsigShifted    
  out asig  
endin

</CsInstruments>
<CsScore>
i $INSTR 0 60
</CsScore>
</CsoundSynthesizer>
