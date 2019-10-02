<CsoundSynthesizer>
<CsOptions>
--opcode-lib=../../libcudapvs2.dylib
--opcode-lib=./libcudapvsmorph2.so
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 128
0dbfs = 1

gifftsize = $FFT
gihopsize = $HOP

instr 4
  asig1 soundin "syrinx.wav"
  asig2 soundin "ends.wav"
  kampint	linseg		0, p3, 1
  kfrqint	linseg		0, p3, 1
  fsig1 cudanal2 asig1, gifftsize, gihopsize, gifftsize, 1
  fsig2 cudanal2 asig2, gifftsize, gihopsize, gifftsize, 1
  ftps cudapvsmorph2 fsig1, fsig2, kampint, kfrqint 
  atps cudasynth2 ftps			                   
  out atps
endin

</CsInstruments>
<CsScore>
i $INSTR 0 60
</CsScore>
</CsoundSynthesizer>
