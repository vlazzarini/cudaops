<CsoundSynthesizer>
<CsOptions>
--opcode-lib=libcudapvsmooth.so
</CsOptions>
<CsInstruments>
ksmps = 128
0dbfs  = 1

instr 1
  ifftsize = 2048
  ihopsize = 512
  kacf = p4
  kfcf = p5
  asig soundin "syrinx.wav"
  fsig = pvsanal(asig, 
               ifftsize, 
               ihopsize, 
               ifftsize, 1)
  ftps cudapvsmooth fsig, kacf, kfcf
  atps pvsynth ftps			; synthesise it                      
  out atps
endin


instr 2
  ifftsize = 2048
  ihopsize = 512
  kacf = p4
  kfcf = p5
  asig soundin "syrinx.wav"
  fsig = pvsanal(asig, 
               ifftsize, 
               ihopsize, 
               ifftsize, 1)
  ftps pvsmooth fsig, kacf, kfcf
  atps pvsynth ftps			; synthesise it                      
  out atps
endin

</CsInstruments>
<CsScore>
;       amp  freq 
i 1 0 3 0.01 0.01	;smooth amplitude and frequency with cutoff frequency of filter at 1% of 1/2 frame-rate (ca 0.86 Hz)
i 2 3 3 0.01 0.01	;smooth amplitude and frequency with cutoff frequency of filter at 1% of 1/2 frame-rate (ca 0.86 Hz)
i 1 6 3  1   0.01	;no smoothing on amplitude, but frequency with cf at 1% of 1/2 frame-rate (ca 0.86 Hz)
i 2 9 3  1   0.01	;no smoothing on amplitude, but frequency with cf at 1% of 1/2 frame-rate (ca 0.86 Hz)
i 1 12 3 .001  1   	;smooth amplitude with cf at 0.1% of 1/2 frame-rate (ca 0.086 Hz)
			;and no smoothing of frequency
i 2 15 3 .001  1	        ;smooth amplitude with cf at 0.1% of 1/2 frame-rate (ca 0.086 Hz)
			;and no smoothing of frequency

e
</CsScore>
</CsoundSynthesizer>
