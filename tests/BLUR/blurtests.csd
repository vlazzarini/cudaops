<CsoundSynthesizer>
<CsOptions>
--opcode-lib=./libcudapvsblur.so
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 128
0dbfs = 1

gifftsize = $FFT
gihopsize = $HOP

instr 1
  asig soundin "syrinx.wav"
  kblurtime line 0, 60, .99
  fsig = pvsanal(asig, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  ftps pvsblur fsig, kblurtime, 1
  atps pvsynth ftps			                   
  out atps
endin

instr 2
  asig soundin "syrinx.wav"
  kblurtime line 0, 60, .99
  fsig = pvsanal(asig, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  ftps cudapvsblur fsig, kblurtime, 1
  atps pvsynth ftps			                   
  out atps
endin

instr 3
  asig soundin "syrinx.wav"
  kblurtime line 0, 60, .99
  fsig = cudanal(asig, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  ftps cudapvsblur fsig, kblurtime, 1
  atps cudasynth ftps			                   
  out atps
endin


</CsInstruments>
<CsScore>
i $INSTR 0 60
e
</CsScore>
</CsoundSynthesizer>
