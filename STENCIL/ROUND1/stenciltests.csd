<CsoundSynthesizer>
<CsOptions>
--opcode-lib=./libcudapvstencil.so
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 128
0dbfs = 1

gifftsize = $FFT
gihopsize = $HOP

instr 1
  kgain linseg 0, p3, 1.5
  klevel linseg 0, p3, .5
  asig1 soundin "ends.wav"
  fsig1 = pvsanal(asig1, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  ftps pvstencil fsig1, kgain, klevel, 1
  atps pvsynth ftps			                   
  out atps
endin

instr 2
  kgain linseg 0, p3, 1.5
  klevel linseg 0, p3, .5
  asig1 soundin "ends.wav"
  fsig1 = pvsanal(asig1, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  ftps cudapvstencil fsig1, kgain, klevel, 1
  atps pvsynth ftps			                   
  out atps
endin

instr 3
  kgain linseg 0, p3, 1.5
  klevel linseg 0, p3, .5
  asig1 soundin "ends.wav"
  fsig1 = cudanal(asig1, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  ftps cudapvstencil fsig1, kgain, klevel, 1
  atps cudasynth ftps			                   
  out atps
endin

</CsInstruments>
<CsScore>
f1 0 -[$FFT+1] 21 1
i $INSTR 0 60
</CsScore>
</CsoundSynthesizer>
