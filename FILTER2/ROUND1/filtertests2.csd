<CsoundSynthesizer>
<CsOptions>
--opcode-lib=./libcudapvsfilter2.so
--opcode-lib=../../libcudapvs2.dylib
</CsOptions>
<CsInstruments>

ksmps = 128
0dbfs = 1

giSine ftgen 0, 0, 4096, 10, 1

gifftsize = $FFT
gihopsize = $HOP

instr 4
  kfreq  expseg 500, p3/3, 4000, p3/3, 500, p3/3, 4000   ; 3-octave sweep
  kdepth linseg 1, p3/2, 0.5, p3/2, 1   ; varying filter depth

  asig soundin "syrinx.wav"           ; input
  afil  oscili  1, kfreq, giSine        ; filter t-domain signal

  fim   cudanal2  asig,gifftsize,gihopsize,gifftsize,0  ; pvoc analysis 
  fil   cudanal2  afil,gifftsize,gihopsize,gifftsize,0  
  fou   cudapvsfilter2 fim, fil, kdepth    
  aout  cudasynth2  fou                   ; pvoc synthesis
  out aout
endin

</CsInstruments>

<CsScore>
i $INSTR 0 60
</CsScore>

</CsoundSynthesizer>
