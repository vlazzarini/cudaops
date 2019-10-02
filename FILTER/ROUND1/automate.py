import os
import sys

f = open("timing.txt", "w")
f.write("%s\n\n" % sys.argv[1])
f.close()


hopsize = ["128","256"]
fftsize = ["1024","2048","4096","8192","16384"]
instrument = ["1","2","3"]
for hop in hopsize:
    for fft in fftsize:
        for instr in instrument:
            print("\nhopsize %s : fftsize %s : instrument %s :" % (hop,fft,instr))
            result = 0
            for exp in range(0,5):
                os.system("csound %s --smacro:INSTR=%s --omacro:HOP=%s --omacro:FFT=%s -n 2> log.txt" % (sys.argv[1],instr,hop,fft))
                f = open("log.txt", "r")
                strc = "Elapsed time at end of performance: real: "
                strt = len(strc)
                timing = 0.0
                for line in f:
                    if(line[0:strt] == strc):
                        end = line[strt:].index(',')
                        timing = float(line[strt:strt+end-1])
                f.close()
                print(timing)
                result += (timing)
            mean = result/5
            f = open("timing.txt", "a")
            f.write("hopsize %s : fftsize %s : instrument %s : %.3f\n" % (hop,fft,instr,mean))
            f.close()

hopsize = ["512"]
fftsize = ["2048","4096","8192","16384"]
instrument = ["1","2","3"]
for hop in hopsize:
    for fft in fftsize:
        for instr in instrument:
            print("\nhopsize %s : fftsize %s : instrument %s :" % (hop,fft,instr))
            result = 0
            for exp in range(0,5):
                os.system("csound %s --smacro:INSTR=%s --omacro:HOP=%s --omacro:FFT=%s -n 2> log.txt" % (sys.argv[1],instr,hop,fft))
                f = open("log.txt", "r")
                strc = "Elapsed time at end of performance: real: "
                strt = len(strc)
                timing = 0.0
                for line in f:
                    if(line[0:strt] == strc):
                        end = line[strt:].index(',')
                        timing = float(line[strt:strt+end-1])
                f.close()
                print(timing)
                result += (timing)
            mean = result/5
            f = open("timing.txt", "a")
            f.write("hopsize %s : fftsize %s : instrument %s : %.3f\n" % (hop,fft,instr,mean))
            f.close()

hopsize = ["1024"]
fftsize = ["4096","8192","16384"]
instrument = ["1","2","3"]
for hop in hopsize:
    for fft in fftsize:
        for instr in instrument:
            print("\nhopsize %s : fftsize %s : instrument %s :" % (hop,fft,instr))
            result = 0
            for exp in range(0,5):
                os.system("csound %s --smacro:INSTR=%s --omacro:HOP=%s --omacro:FFT=%s -n 2> log.txt" % (sys.argv[1],instr,hop,fft))
                f = open("log.txt", "r")
                strc = "Elapsed time at end of performance: real: "
                strt = len(strc)
                timing = 0.0
                for line in f:
                    if(line[0:strt] == strc):
                        end = line[strt:].index(',')
                        timing = float(line[strt:strt+end-1])
                f.close()
                print(timing)
                result += (timing)
            mean = result/5
            f = open("timing.txt", "a")
            f.write("hopsize %s : fftsize %s : instrument %s : %.3f\n" % (hop,fft,instr,mean))
            f.close()

hopsize = ["2048"]
fftsize = ["8192","16384"]
instrument = ["1","2","3"]
for hop in hopsize:
    for fft in fftsize:
        for instr in instrument:
            print("\nhopsize %s : fftsize %s : instrument %s :" % (hop,fft,instr))
            result = 0
            for exp in range(0,5):
                os.system("csound %s --smacro:INSTR=%s --omacro:HOP=%s --omacro:FFT=%s -n 2> log.txt" % (sys.argv[1],instr,hop,fft))
                f = open("log.txt", "r")
                strc = "Elapsed time at end of performance: real: "
                strt = len(strc)
                timing = 0.0
                for line in f:
                    if(line[0:strt] == strc):
                        end = line[strt:].index(',')
                        timing = float(line[strt:strt+end-1])
                f.close()
                print(timing)
                result += (timing)
            mean = result/5
            f = open("timing.txt", "a")
            f.write("hopsize %s : fftsize %s : instrument %s : %.3f\n" % (hop,fft,instr,mean))
            f.close()

hopsize = ["4096"]
fftsize = ["16384"]
instrument = ["1","2","3"]
for hop in hopsize:
    for fft in fftsize:
        for instr in instrument:
            print("\nhopsize %s : fftsize %s : instrument %s :" % (hop,fft,instr))
            result = 0
            for exp in range(0,5):
                os.system("csound %s --smacro:INSTR=%s --omacro:HOP=%s --omacro:FFT=%s -n 2> log.txt" % (sys.argv[1],instr,hop,fft))
                f = open("log.txt", "r")
                strc = "Elapsed time at end of performance: real: "
                strt = len(strc)
                timing = 0.0
                for line in f:
                    if(line[0:strt] == strc):
                        end = line[strt:].index(',')
                        timing = float(line[strt:strt+end-1])
                f.close()
                print(timing)
                result += (timing)
            mean = result/5
            f = open("timing.txt", "a")
            f.write("hopsize %s : fftsize %s : instrument %s : %.3f\n" % (hop,fft,instr,mean))
            f.close()
