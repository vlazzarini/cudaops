<CsoundSynthesizer>
<CsOptions>
--opcode-lib=../../libcudapvs2.dylib
--opcode-lib=./libcudapvsmix2.so
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
  fsig1 cudanal2 asig1, gifftsize, gihopsize, gifftsize, 1
  fsig2 cudanal2 asig2, gifftsize, gihopsize, gifftsize, 1
  ftps cudapvsmix2 fsig1, fsig2
  atps cudasynth2 ftps			                   
  out atps
endin

</CsInstruments>
<CsScore>
i $INSTR 0 60
</CsScore>
</CsoundSynthesizer>
