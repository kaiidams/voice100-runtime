CFLAGS = -I.
LDFLAGS = -L.

run: test
	LD_LIBRARY_PATH=. ./test

test: test.cpp
	$(CC) $(CFLAGS) test.cpp -o test $(LDFLAGS) -lWORLD -lstdc++ -lm
