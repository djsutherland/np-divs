CPP      = clang++
CPPFLAGS = -Wall -g
LDFLAGS  = -lflann
EIGEN    = /usr/local/include/eigen3/

# general rule for how to compile a .o file:
%.o: %.cpp
	$(CPP) -c $(CPPFLAGS) -I$(EIGEN) $<

.PHONY: all clean cleanest
all: np_divs
div_funcs = div_func.o div_l2.o div_alpha.o div_renyi.o div_bc.o \
			div_hellinger.o utils.o

################################################################################
### Divergence estimator

np_divs: np_divs.o $(div_funcs) gamma.o
	$(CPP) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

np_divs.o: np_divs.hpp div_l2.hpp
np_divs.hpp: div_func.hpp

################################################################################
### Divergence functions

div_func.o: div_func.hpp

div_l2.o: div_l2.hpp utils.hpp gamma.hpp
div_l2.hpp: div_func.hpp

div_alpha.o: div_alpha.hpp utils.hpp gamma.hpp
div_alpha.hpp: div_func.hpp

div_renyi.o: div_renyi.hpp
div_renyi.hpp: div_alpha.hpp

div_bc.o: div_bc.hpp
div_bc.hpp: div_alpha.hpp

div_hellinger.o: div_bc.hpp
div_hellinger.hpp: div_alpha.hpp

################################################################################
### Utilities

gamma.o: gamma.hpp
utils.o: utils.hpp

################################################################################
### Cleanup
clean:
	rm -f *.o

cleanest: clean
	rm -f np_divs
