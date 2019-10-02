<CsoundSynthesizer>
<CsOptions>
--opcode-lib=../../libcudapvs2.dylib
--opcode-lib=./libcudapvsblur2.so
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 128
0dbfs = 1

gifftsize = $FFT
gihopsize = $HOP

instr 4
  asig soundin "syrinx.wav"
  kblurtime line 0, 60, .99
  fsig cudanal2 asig, gifftsize, gihopsize, gifftsize, 1
  ftps cudapvsblur2 fsig, kblurtime, 1
  atps cudasynth2 ftps			                   
  out atps
endin


</CsInstruments>
<CsScore>
i $INSTR 0 60
e
</CsScore>
</CsoundSynthesizer>
