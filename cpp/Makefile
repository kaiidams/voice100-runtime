CFLAGS = -I../src
LDFLAGS = -L../build/x86_64-linux-gnu

run: test
	LD_LIBRARY_PATH=. ./test

test: test.cpp
	$(CC) $(CFLAGS) test.cpp -o test $(LDFLAGS) -lWORLD -lstdc++ -lm
