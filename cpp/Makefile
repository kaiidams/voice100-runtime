CFLAGS = -I.
LDFLAGS = -L.
LIBS = -lWORLD -lstdc++ -lm

all: test
	LD_LIBRARY_PATH=. ./test

test: test.o vocoder.o
	$(CC) test.o vocoder.o -o $@ $(LDFLAGS) $(LIBS)

test.o: test.cpp
	$(CC) -c $(CFLAGS) $< -o $@
	
vocoder.o: vocoder.cpp
	$(CC) -c $(CFLAGS) $< -o $@
	
