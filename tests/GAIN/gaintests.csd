<CsoundSynthesizer>
<CsOptions>
--opcode-lib=./libcudapvsgain.so
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 128
0dbfs = 1

gifftsize = $FFT
gihopsize = $HOP

instr 1
  kenv linseg 0, p3/4, 1, p3/4, 0, p3/4, 1.5, p3/4, 0
  asig soundin "syrinx.wav"
  fsig = pvsanal(asig, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  fsigScaled pvsgain fsig, kenv
  asig pvsynth fsigScaled    
  out asig
endin

instr 2
  kenv linseg 0, p3/4, 1, p3/4, 0, p3/4, 1.5, p3/4, 0
  asig soundin "syrinx.wav"
  fsig = pvsanal(asig, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  fsigScaled cudapvsgain fsig, kenv
  asig pvsynth fsigScaled    
  out asig
endin

instr 3
  kenv linseg 0, p3/4, 1, p3/4, 0, p3/4, 1.5, p3/4, 0
  asig soundin "syrinx.wav"
  fsig = cudanal(asig, 
               gifftsize, 
               gihopsize, 
               gifftsize, 1)
  fsigScaled cudapvsgain fsig, kenv
  asig cudasynth fsigScaled    
  out asig
endin

</CsInstruments>
<CsScore>
i $INSTR 0 60
</CsScore>
</CsoundSynthesizer>
