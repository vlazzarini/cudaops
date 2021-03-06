<CsoundSynthesizer>
<CsOptions>
--opcode-lib=./libcudapvsfilter.so
</CsOptions>
<CsInstruments>

ksmps = 128
0dbfs = 1

giSine ftgen 0, 0, 4096, 10, 1

gifftsize = $FFT
gihopsize = $HOP

instr 1
  kfreq  expseg 500, p3/3, 4000, p3/3, 500, p3/3, 4000   ; 3-octave sweep
  kdepth linseg 1, p3/2, 0.5, p3/2, 1   ; varying filter depth

  asig soundin "syrinx.wav"           ; input
  afil  oscili  1, kfreq, giSine        ; filter t-domain signal

  fim   pvsanal  asig,gifftsize,gihopsize,gifftsize,0  ; pvoc analysis 
  fil   pvsanal  afil,gifftsize,gihopsize,gifftsize,0  
  fou   pvsfilter fim, fil, kdepth    
  aout  pvsynth  fou                   ; pvoc synthesis
  out(aout)
endin

instr 2
  kfreq  expseg 500, p3/3, 4000, p3/3, 500, p3/3, 4000   ; 3-octave sweep
  kdepth linseg 1, p3/2, 0.5, p3/2, 1   ; varying filter depth

  asig soundin "syrinx.wav"           ; input
  afil  oscili  1, kfreq, giSine        ; filter t-domain signal

  fim   pvsanal  asig,gifftsize,gihopsize,gifftsize,0  ; pvoc analysis 
  fil   pvsanal  afil,gifftsize,gihopsize,gifftsize,0  
  fou   cudapvsfilter fim, fil, kdepth    
  aout  pvsynth  fou                   ; pvoc synthesis
  out(aout)
endin

instr 3
  kfreq  expseg 500, p3/3, 4000, p3/3, 500, p3/3, 4000   ; 3-octave sweep
  kdepth linseg 1, p3/2, 0.5, p3/2, 1   ; varying filter depth

  asig soundin "syrinx.wav"           ; input
  afil  oscili  1, kfreq, giSine        ; filter t-domain signal

  fim   cudanal  asig,gifftsize,gihopsize,gifftsize,0  ; pvoc analysis 
  fil   cudanal  afil,gifftsize,gihopsize,gifftsize,0  
  fou   cudapvsfilter fim, fil, kdepth    
  aout  cudasynth  fou                   ; pvoc synthesis
  out(aout)
endin

</CsInstruments>

<CsScore>
i $INSTR 0 60
</CsScore>

</CsoundSynthesizer>
