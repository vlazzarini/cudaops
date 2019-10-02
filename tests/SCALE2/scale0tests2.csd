<CsoundSynthesizer>
<CsOptions>
--opcode-lib=../../libcudapvs2.dylib
--opcode-lib=libcudapvscale2.so
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 128
0dbfs = 1

gifftsize = $FFT
gihopsize = $HOP

instr 4
  kscale linseg .3, p3, 3
  kgain linseg 1, p3/3, 0, p3/3, 1.5, p3/3, 1 
  asig soundin "syrinx.wav"
  fsig cudanal2 asig, gifftsize, gihopsize, gifftsize, 1
  fsigScaled cudapvscale2 fsig, kscale, 0, kgain
  asig cudasynth2 fsigScaled     
  out asig
endin

</CsInstruments>
<CsScore>
i $INSTR 0 60
</CsScore>
</CsoundSynthesizer>
