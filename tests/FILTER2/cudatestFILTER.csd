<CsoundSynthesizer>
<CsOptions>
--opcode-lib=libcudapvsfilter.so
</CsOptions>
<CsInstruments>

ksmps = 128
0dbfs = 1

giSine ftgen 0, 0, 4096, 10, 1

instr 1
  ifftsize = 2048
  ihopsize = 512
  kfreq  expon 500, p3, 4000           ; 3-octave sweep
  ;kdepth linseg 1, p3/2, 0.5, p3/2, 1  ; varying filter depth
  kdepth = .9

  asig = diskin2:a("ends.wav",1,0,1)   ; input
  afil  oscili  1, kfreq, giSine        ; filter t-domain signal

  fim   pvsanal  asig,1024,256,1024,0  ; pvoc analysis 
  fil   pvsanal  afil,1024,256,1024,0  
  fou   cudapvsfilter fim, fil, kdepth    
  aout  pvsynth  fou                   ; pvoc synthesis
  out(aout)
endin

instr 2
  ifftsize = 2048
  ihopsize = 512
  kfreq  expon 500, p3, 4000           ; 3-octave sweep
  ;kdepth linseg 1, p3/2, 0.5, p3/2, 1  ; varying filter depth
  kdepth = .9

  asig = diskin2:a("ends.wav",1,0,1)   ; input
  afil  oscili  1, kfreq, giSine        ; filter t-domain signal

  fim   pvsanal  asig,1024,256,1024,0  ; pvoc analysis
  fil   pvsanal  afil,1024,256,1024,0  
  fou   pvsfilter fim, fil, kdepth     
  aout  pvsynth  fou                   ; pvoc synthesis
  out(aout)
endin

</CsInstruments>

<CsScore>
i1 0 5
i2 5.1 5
</CsScore>

</CsoundSynthesizer>
