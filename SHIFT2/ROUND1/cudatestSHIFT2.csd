<CsoundSynthesizer>
<CsOptions>
--opcode-lib=../../libcudapvs2.dylib
--opcode-lib=libcudapvshift2.so
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
fsigShifted pvshift fsig,kshift,klowest,0,1
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
fsig cudanal2 asig, ifftsize, ihopsize, ifftsize, 1
fsigShifted cudapvshift2 fsig,kshift,klowest,0,1
asig cudasynth2 fsigShifted    
   out asig
endin


</CsInstruments>
<CsScore>

i1 0 5
i2 5.1 5
</CsScore>
</CsoundSynthesizer>
