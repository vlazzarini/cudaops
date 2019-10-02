<CsoundSynthesizer>
<CsOptions>
--opcode-lib=libcudapvsgain.so
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 128
0dbfs = 1

gifftsize = 2048
gihopsize = 512

instr 1
  kgain linseg 0, p3/3, 1.5, p3/3, 0, p3/3, 1
  asig soundin "syrinx.wav"
  fsig = pvsanal(asig, gifftsize, gihopsize, gifftsize, 1)
  fsigScaled cudapvsgain fsig, kgain
  asig pvsynth fsigScaled     
  out(asig)
endin

</CsInstruments>
<CsScore>
i 1 0 5
</CsScore>
</CsoundSynthesizer>
