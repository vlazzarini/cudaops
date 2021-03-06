<CsoundSynthesizer>
<CsOptions>
--opcode-lib=./libcudapvsmorph.so
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 128
0dbfs = 1

gifftsize = $FFT
gihopsize = $HOP

instr 1
  asig1 soundin "syrinx.wav"
  asig2 soundin "ends.wav"
  kampint	linseg		0, p3, 1
  kfrqint	linseg		0, p3, 1
  fsig1 = pvsanal(asig1, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  fsig2 = pvsanal(asig2, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  ftps pvsmorph fsig1, fsig2, kampint, kfrqint 
  atps pvsynth ftps			                   
  out atps
endin

instr 2
  asig1 soundin "syrinx.wav"
  asig2 soundin "ends.wav"
  kampint	linseg		0, p3, 1
  kfrqint	linseg		0, p3, 1
  fsig1 = pvsanal(asig1, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  fsig2 = pvsanal(asig2, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  ftps cudapvsmorph fsig1, fsig2, kampint, kfrqint 
  atps pvsynth ftps			                   
  out atps
endin

instr 3
  asig1 soundin "syrinx.wav"
  asig2 soundin "ends.wav"
  kampint	linseg		0, p3, 1
  kfrqint	linseg		0, p3, 1
  fsig1 = cudanal(asig1, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  fsig2 = cudanal(asig2, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  ftps cudapvsmorph fsig1, fsig2, kampint, kfrqint 
  atps cudasynth ftps			                   
  out atps
endin

</CsInstruments>
<CsScore>
i $INSTR 0 60
</CsScore>
</CsoundSynthesizer>
