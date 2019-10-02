<CsoundSynthesizer>
<CsOptions>
--opcode-lib=../../libcudapvs2.dylib --opcode-lib=./libcudapvsgain2.so
</CsOptions>
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 128
0dbfs = 1

gifftsize = 2048
gihopsize = 512

instr 1
  kgain linseg 1.5, p3/2, 0, p3/2, 1
  asig soundin "syrinx.wav"
  fsig cudanal2 asig, gifftsize, gihopsize, gifftsize, 1
  fsigScaled cudapvsgain2 fsig, kgain
  asig cudasynth2 fsigScaled   
  out(asig)
endin

</CsInstruments>
<CsScore>
i 1 0 60
</CsScore>
</CsoundSynthesizer>
