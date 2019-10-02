<CsoundSynthesizer>
<CsOptions>
--opcode-lib=libcudapvsmix.so
</CsOptions>
<CsInstruments>

ksmps = 128
0dbfs  = 1

instr 1
  ifftsize = 2048
  ihopsize = 512
  asig1 soundin "syrinx.wav"
  asig2 soundin "ends.wav"
  fsig1 = pvsanal(asig1, 
               ifftsize, 
               ihopsize, 
               ifftsize, 1)
  fsig2 = pvsanal(asig2, 
               ifftsize, 
               ihopsize, 
               ifftsize, 1)
  ftps cudapvsmix fsig1, fsig2
  atps pvsynth ftps			                   
  out atps
endin


instr 2
  ifftsize = 2048
  ihopsize = 512
  asig1 soundin "syrinx.wav"
  asig2 soundin "ends.wav"
  fsig1 = pvsanal(asig1, 
               ifftsize, 
               ihopsize, 
               ifftsize, 1)
  fsig2 = pvsanal(asig2, 
               ifftsize, 
               ihopsize, 
               ifftsize, 1)
  ftps pvsmix fsig1, fsig2
  atps pvsynth ftps			                   
  out atps
endin

</CsInstruments>
<CsScore>
i1 0  4 
i2 5  4

e
</CsScore>
</CsoundSynthesizer>
