<CsoundSynthesizer>
<CsOptions>
--opcode-lib=../../libcudapvs2.dylib
--opcode-lib=./libcudapvscale2.so
</CsOptions>
<CsInstruments>

ksmps = 128
0dbfs = 1

instr 1
ifftsize = 2048
ihopsize = 512
;kscale linseg .4, 4, 2
kscale = .75
asig = .5 * diskin2:a("brian.wav",1,0,1)
fsig = pvsanal(asig, 
               ifftsize, 
               ihopsize, 
               ifftsize, 1)
fsigScaled pvscale fsig,kscale,1,.7 
asig = pvsynth(fsigScaled)    
   out(asig)
endin

instr 2
ifftsize = 2048
ihopsize = 512
;kscale linseg .4, 4, 2
kscale = .75
asig = .5 * diskin2:a("brian.wav",1,0,1)
fsig cudanal2 asig, ifftsize, ihopsize, ifftsize, 1
fsigScaled cudapvscale2 fsig,kscale,1,.7
asig cudasynth2 fsigScaled   
   out(asig)
endin

instr 3
ifftsize = 2048
ihopsize = 512
asig = diskin2:a("syrinx.wav",1,0,1)   
   out(asig*.6)
endin


</CsInstruments>
<CsScore>

i1 0 3
i2 3.1 3
; i2 6.2 3
</CsScore>
</CsoundSynthesizer>
