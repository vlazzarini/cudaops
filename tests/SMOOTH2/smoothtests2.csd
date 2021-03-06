<CsoundSynthesizer>
<CsOptions>
--opcode-lib=../../libcudapvs2.dylib
--opcode-lib=./libcudapvsmooth2.so
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 128
0dbfs = 1

gifftsize = $FFT
gihopsize = $HOP

instr 4
  kacf linseg 0.001, p3, 1
  kfcf linseg 1, p3, 0.001
  asig soundin "syrinx.wav"
  fsig cudanal2 asig, gifftsize, gihopsize, gifftsize, 1
  ftps cudapvsmooth2 fsig, kacf, kfcf
  atps cudasynth2 ftps			; synthesise it                      
  out atps
endin

</CsInstruments>
<CsScore>
i $INSTR 0 60
</CsScore>
</CsoundSynthesizer>
