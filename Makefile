CC      = g++
CFLAGS  = -I/usr/local/include/eigen3/
LDFLAGS = -lflann

all: div_l2.o

# np_divs: np_divs.o
# 	$(CC) -o $@ $^ $(LDFLAGS)
# 
# np_divs.o: np_divs.cpp np_divs.hpp
# 	$(CC) -c $(CFLAGS) $<

div_l2.o: div_l2.cpp div_l2.hpp div_func.o gamma_half.o utils.o
	$(CC) -c $(CFLAGS) $<

div_l2.hpp: div_func.hpp

div_func.o: div_func.cpp div_func.hpp
	$(CC) -c $(CFLAGS) $<

gamma_half.o: gamma_half.cpp gamma_half.hpp
	$(CC) -c $(CFLAGS) $<

utils.o: utils.cpp utils.hpp
	$(CC) -c $(CFLAGS) $<

.PHONY: clean cleaneanest

clean:
	rm -f *.o

cleanest: clean
	rm -f np_divs
