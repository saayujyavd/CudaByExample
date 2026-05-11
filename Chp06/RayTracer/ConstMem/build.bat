cls

del *.exp
del *.lib
del *.exe

nvcc main.cu -o out.exe -allow-unsupported-compiler

del *.exp
del *.lib
