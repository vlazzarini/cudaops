<CsoundSynthesizer>
<CsOptions>
--opcode-lib=../../libcudapvs2.dylib
--opcode-lib=./libcudapvstencil2.so
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 128
0dbfs = 1

gifftsize = $FFT
gihopsize = $HOP

instr 4
  kgain linseg 0, p3, 1.5
  klevel linseg 0, p3, .5
  asig1 soundin "ends.wav"
  fsig1 cudanal2 asig1, gifftsize, gihopsize, gifftsize, 1
  ftps cudapvstencil2 fsig1, kgain, klevel, 1
  atps cudasynth2 ftps			                   
  out atps
endin

</CsInstruments>
<CsScore>
f1 0 -[$FFT+1] 21 1
i $INSTR 0 60
</CsScore>
</CsoundSynthesizer>
