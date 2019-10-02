<CsoundSynthesizer>
<CsOptions>
--opcode-lib=libcudapvshift.so
</CsOptions>
<CsInstruments>

ksmps = 128
0dbfs = 1

instr 1
ifftsize = 2048
ihopsize = 512
;kscale linseg .4, 4, 2
kshift = 300
klowest = 0
asig = diskin2:a("syrinx.wav",1,0,1)
fsig = pvsanal(asig, 
               ifftsize, 
               ihopsize, 
               ifftsize, 1)
fsigShifted cudapvshift fsig,kshift,klowest,0,1
asig = pvsynth(fsigShifted)    
   out(asig)
endin

instr 2
ifftsize = 2048
ihopsize = 512
;kscale linseg .4, 4, 2
kshift = 300
klowest = 0
asig = diskin2:a("syrinx.wav",1,0,1)
fsig = pvsanal(asig, 
               ifftsize, 
               ihopsize, 
               ifftsize, 1)
fsigShifted pvshift fsig,kshift,klowest,0,1
asig = pvsynth(fsigShifted)    
   out(asig)
endin

instr 3
ifftsize = 2048
ihopsize = 512
asig = diskin2:a("brian.wav",1,0,1)   
   out(asig)
endin


</CsInstruments>
<CsScore>

i1 0 5
i2 5.1 5
</CsScore>
</CsoundSynthesizer>
