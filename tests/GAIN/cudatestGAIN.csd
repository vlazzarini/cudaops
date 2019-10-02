<CsoundSynthesizer>
<CsOptions>
--opcode-lib=libcudaop6.so
</CsOptions>
<CsInstruments>

ksmps = 64
0dbfs = 1

instr 1
ifftsize = 2048
ihopsize = 512
kenv linsegr 0, p3/2, 1, p3/4,1,p3/4,0
asig = diskin2:a("beats.wav",1,0,1)
fsig = cudanal(asig, 
               ifftsize, 
               ihopsize, 
               ifftsize, 1)
;fsigScaled = cudapvsgain(fsig,kenv)
fsigScaled = pvsgain(fsig,kenv)
asig = cudasynth(fsigScaled)    
   out(asig)
endin

instr 2
ifftsize = 2048
ihopsize = 512
kenv linsegr 0, p3/2, 1, p3/4,1,p3/4,0
asig = diskin2:a("beats.wav",1,0,1)
fsig = pvsanal(asig, 
               ifftsize, 
               ihopsize, 
               ifftsize, 1)
fsigScaled = pvsgain(fsig,kenv)
asig = pvsynth(fsigScaled)    
   out(asig)
endin

instr 3
ifftsize = 2048
ihopsize = 512
kenv linsegr 0, p3/2, 1, p3/4,1,p3/4,0
asig = diskin2:a("beats.wav",1,0,1)
fsig = pvsanal(asig, 
               ifftsize, 
               ihopsize, 
               ifftsize, 1)
fsigScaled = cudapvsgain(fsig,kenv)
asig = pvsynth(fsigScaled)    
   out(asig)
endin

</CsInstruments>
<CsScore>
i3 0 60
</CsScore>
</CsoundSynthesizer>
