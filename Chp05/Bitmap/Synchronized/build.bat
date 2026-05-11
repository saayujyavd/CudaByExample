cls

del *.exp
del *.lib
del *.exe

nvcc sync.cu -o out.exe -allow-unsupported-compiler

del *.exp
del *.lib
