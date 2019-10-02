<CsoundSynthesizer>
<CsOptions>
--opcode-lib=libcudapvsmorph.so
</CsOptions>
<CsInstruments>
ksmps = 128
0dbfs  = 1

instr 1
  ifftsize = 2048
  ihopsize = 512
  asig1 soundin "syrinx.wav"
  asig2 soundin "ends.wav"
  iampint1	=		p4
  iampint2	=		p5
  ifrqint1	=		p6
  ifrqint2	=		p7
  kampint	linseg		iampint1, p3, iampint2
  kfrqint	linseg		ifrqint1, p3, ifrqint2
  fsig1 = pvsanal(asig1, 
               ifftsize, 
               ihopsize, 
               ifftsize, 1)
  fsig2 = pvsanal(asig2, 
               ifftsize, 
               ihopsize, 
               ifftsize, 1)
  ftps cudapvsmorph fsig1, fsig2, kampint, kfrqint 
  atps pvsynth ftps			                   
  out atps
endin


instr 2
  ifftsize = 2048
  ihopsize = 512
  asig1 soundin "syrinx.wav"
  asig2 soundin "ends.wav"
  iampint1	=		p4
  iampint2	=		p5
  ifrqint1	=		p6
  ifrqint2	=		p7
  kampint	linseg		iampint1, p3, iampint2
  kfrqint	linseg		ifrqint1, p3, ifrqint2
  fsig1 = pvsanal(asig1, 
               ifftsize, 
               ihopsize, 
               ifftsize, 1)
  fsig2 = pvsanal(asig2, 
               ifftsize, 
               ihopsize, 
               ifftsize, 1)
  ftps pvsmorph fsig1, fsig2, kampint, kfrqint 
  atps pvsynth ftps			                   
  out atps
endin

</CsInstruments>
<CsScore>
i1 0  3 0 0 1 1
i2 4  3 0 0 1 1
i1 8  3 1 0 1 0
i2 12 3 1 0 1 0
i1 16 3 0 1 0 1
i2 20 3 0 1 0 1
e
</CsScore>
</CsoundSynthesizer>
