<CsoundSynthesizer>
<CsOptions>
--opcode-lib=libcudapvscale.so
</CsOptions>
<CsInstruments>

ksmps = 128
0dbfs = 1

instr 1
ifftsize = 2048
ihopsize = 512
;kscale linseg .4, 4, 2
kscale = .5
asig = diskin2:a("syrinx.wav",1,0,1)
fsig = pvsanal(asig, 
               ifftsize, 
               ihopsize, 
               ifftsize, 1)
fsigScaled cudapvscale fsig,kscale,0,.7 
asig = pvsynth(fsigScaled)    
   out(asig)
endin

instr 2
ifftsize = 2048
ihopsize = 512
;kscale linseg .4, 4, 2
kscale = .5
asig = diskin2:a("syrinx.wav",1,0,1)
fsig = pvsanal(asig, 
               ifftsize, 
               ihopsize, 
               ifftsize, 1)
fsigScaled pvscale fsig,kscale,0,.7
fsigScaled pvsgain fsigScaled, .9
asig = pvsynth(fsigScaled)    
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

i3 0 3
i1 3.1 3
i2 6.2 3
</CsScore>
</CsoundSynthesizer>
