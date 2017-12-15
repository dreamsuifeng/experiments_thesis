#Makefile for univerSVM software (c) 2005 

CXX= g++
MEXFLAGS= -DMATLAB_MEX_FILE -DMEX
USEMEXFLAGS= 
CFLAGS= -O3 
SVQPSRC= svqp2/svqp2.cpp svqp2/vector.c svqp2/messages.c
MEXSRC= svqp2mex.cpp

svqp: $(SVQPSRC)
	$(CXX) $(CFLAGS) $(USEMEXFLAGS) -fPIC -o svqp2/vector.o -c svqp2/vector.c
	$(CXX) $(CFLAGS) $(USEMEXFLAGS) -fPIC -o svqp2/messages.o -c svqp2/messages.c
	$(CXX) $(CFLAGS) $(USEMEXFLAGS) -fPIC -o svqp2/svqp2.o -c svqp2/svqp2.cpp

mex: $(MEXSRC) $(SVQPSRC)
	make "USEMEXFLAGS=$(MEXFLAGS)" svqp
	mex $(MEXFLAGS) $(MEXSRC)  svqp2/svqp2.o svqp2/messages.o svqp2/vector.o
