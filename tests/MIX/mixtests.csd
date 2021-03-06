<CsoundSynthesizer>
<CsOptions>
--opcode-lib=./libcudapvsmix.so
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
  fsig1 = pvsanal(asig1, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  fsig2 = pvsanal(asig2, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  ftps pvsmix fsig1, fsig2
  atps pvsynth ftps			                   
  out atps
endin

instr 2
  asig1 soundin "syrinx.wav"
  asig2 soundin "ends.wav"
  fsig1 = pvsanal(asig1, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  fsig2 = pvsanal(asig2, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  ftps cudapvsmix fsig1, fsig2
  atps pvsynth ftps			                   
  out atps
endin

instr 3
  asig1 soundin "syrinx.wav"
  asig2 soundin "ends.wav"
  fsig1 = cudanal(asig1, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  fsig2 = cudanal(asig2, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  ftps cudapvsmix fsig1, fsig2
  atps cudasynth ftps			                   
  out atps
endin

</CsInstruments>
<CsScore>
i $INSTR 0 60
</CsScore>
</CsoundSynthesizer>
