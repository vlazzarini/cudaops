<CsoundSynthesizer>
<CsOptions>
--opcode-lib=./libcudapvsmooth.so
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 128
0dbfs = 1

gifftsize = $FFT
gihopsize = $HOP

instr 1
  kacf linseg 0.001, p3, 1
  kfcf linseg 1, p3, 0.001
  asig soundin "syrinx.wav"
  fsig = pvsanal(asig, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  ftps pvsmooth fsig, kacf, kfcf
  atps pvsynth ftps			; synthesise it                      
  out atps
endin

instr 2
  kacf linseg 0.001, p3, 1
  kfcf linseg 1, p3, 0.001
  asig soundin "syrinx.wav"
  fsig = pvsanal(asig, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  ftps cudapvsmooth fsig, kacf, kfcf
  atps pvsynth ftps			; synthesise it                      
  out atps
endin

instr 3
  kacf linseg 0.001, p3, 1
  kfcf linseg 1, p3, 0.001
  asig soundin "syrinx.wav"
  fsig = cudanal(asig, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  ftps cudapvsmooth fsig, kacf, kfcf
  atps cudasynth ftps			; synthesise it                      
  out atps
endin

</CsInstruments>
<CsScore>
i $INSTR 0 60
</CsScore>
</CsoundSynthesizer>
