<CsoundSynthesizer>
<CsOptions>
--opcode-lib=../../libcudapvs2.dylib 
--opcode-lib=./libcudapvsgain2.so
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 128
0dbfs = 1

gifftsize = $FFT
gihopsize = $HOP

instr 4
  kenv linseg 0, p3/4, 1, p3/4, 0, p3/4, 1.5, p3/4, 0
  asig soundin "syrinx.wav"
  fsig cudanal2 asig, gifftsize, gihopsize, gifftsize, 1
  fsigScaled cudapvsgain2 fsig, kenv
  asig cudasynth2 fsigScaled    
  out asig
endin

</CsInstruments>
<CsScore>
i $INSTR 0 60
</CsScore>
</CsoundSynthesizer>
