<CsoundSynthesizer>
<CsOptions>
--opcode-lib=./libcudapvscale.so
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 128
0dbfs = 1

gifftsize = $FFT
gihopsize = $HOP

instr 1
  kscale linseg .3, p3, 3
  kgain linseg 1, p3/3, 0, p3/3, 1.5, p3/3, 1 
  asig soundin "syrinx.wav"
  fsig = pvsanal(asig, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  fsigScaled pvscale fsig, kscale, 2, kgain
  asig pvsynth fsigScaled     
  out asig
endin

instr 2
  kscale linseg .3, p3, 3
  kgain linseg 1, p3/3, 0, p3/3, 1.5, p3/3, 1 
  asig soundin "syrinx.wav"
  fsig = pvsanal(asig, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  fsigScaled cudapvscale fsig, kscale, 2, kgain
  asig pvsynth fsigScaled     
  out asig
endin

instr 3
  kscale linseg .3, p3, 3
  kgain linseg 1, p3/3, 0, p3/3, 1.5, p3/3, 1 
  asig soundin "syrinx.wav"
  fsig = cudanal(asig, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  fsigScaled cudapvscale fsig, kscale, 2, kgain
  asig cudasynth fsigScaled     
  out asig
endin

</CsInstruments>
<CsScore>
i $INSTR 0 60
</CsScore>
</CsoundSynthesizer>
