CC      = clang++
CFLAGS  = -I/usr/local/include/eigen3/ -Wall
LDFLAGS = -lflann

all: np_divs

np_divs: np_divs.o div_l2.o div_func.o gamma.o
	$(CC) -o $@ $^ $(LDFLAGS)

np_divs.o: np_divs.cpp np_divs.hpp div_l2.hpp
	$(CC) -c $(CFLAGS) $<
np_divs.hpp: div_func.hpp

div_l2.o: div_l2.cpp div_l2.hpp div_func.hpp utils.hpp gamma.hpp
	$(CC) -c $(CFLAGS) $<
div_l2.hpp: div_func.hpp

div_func.o: div_func.cpp div_func.hpp
	$(CC) -c $(CFLAGS) $<

gamma.o: gamma.cpp gamma.hpp
	$(CC) -c $(CFLAGS) $<

.PHONY: clean cleanest

clean:
	rm -f *.o

cleanest: clean
	rm -f np_divs
