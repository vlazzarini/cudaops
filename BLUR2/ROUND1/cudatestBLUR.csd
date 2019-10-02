<CsoundSynthesizer>
<CsOptions>
--opcode-lib=libcudapvsblur.so
</CsOptions>
<CsInstruments>

ksmps = 128
0dbfs  = 1

instr 1
  ifftsize = 2048
  ihopsize = 512
  asig soundin "syrinx.wav"
  fsig = pvsanal(asig, 
               ifftsize, 
               ihopsize, 
               ifftsize, 1)
  ftps cudapvsblur fsig, p4, 1
  atps pvsynth ftps			                   
  out atps
endin


instr 2
  ifftsize = 2048
  ihopsize = 512
  asig soundin "syrinx.wav"
  fsig = pvsanal(asig, 
               ifftsize, 
               ihopsize, 
               ifftsize, 1)
  ftps pvsblur fsig, p4, 1
  atps pvsynth ftps			                   
  out atps
endin

</CsInstruments>
<CsScore>
i 1 0   3 0
i 2 4   3 0
i 1 8   3 .1
i 2 12  3 .1
i 1 16  3 .5
i 2 20  3 .5
e
</CsScore>
</CsoundSynthesizer>
