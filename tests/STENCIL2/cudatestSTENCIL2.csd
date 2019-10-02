<CsoundSynthesizer>
<CsOptions>
--opcode-lib=../../libcudapvs2.dylib
--opcode-lib=libcudapvstencil2.so
</CsOptions>
<CsInstruments>
ksmps = 128
0dbfs  = 1

instr 2
  ifftsize = 2048
  ihopsize = 512
  asig1 soundin "ends.wav"
  fsig1 = pvsanal(asig1, 
               ifftsize, 
               ihopsize, 
               ifftsize, 1)
  ftps pvstencil fsig1, p4, p5, p6 
  atps pvsynth ftps			                   
  out atps
endin

instr 3
  ifftsize = 2048
  ihopsize = 512
  asig1 soundin "ends.wav"
  fsig1 cudanal2 asig1, ifftsize, ihopsize, ifftsize, 1
  ftps cudapvstencil2 fsig1, p4, p5, p6 
  atps cudasynth2 ftps			                   
  out atps
endin

</CsInstruments>
<CsScore>

f1 1.9 -1025 -43 "mypvs.pvx" 0
f2 0   -1025 -5   0.000001 400  1 [1025-400] 0.000001
f3 0   -1025 -7   0      1025 -1
f4 0   -1025 9    1 2 0  3 2 0  9 0.333 180
f5 0   -1025 21   1

i2 2   3  0    1   5
i3 6   3  0    1   5
i2 10  3  .1   1   5
i3 14  3  .1   1   5
i2 18  3  3   0.3   5
i3 22  3  3   0.3   5
e
</CsScore>
</CsoundSynthesizer>
