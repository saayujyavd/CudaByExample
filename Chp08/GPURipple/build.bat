cls

del *.exe
del *.lib
del *.exp

nvcc ripple.cu -o out.exe -allow-unsupported-compiler

del *.lib
del *.exp
