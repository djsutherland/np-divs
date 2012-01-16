CPP      = clang++
CPPFLAGS = -Wall -g
LDFLAGS  = -lflann
EIGEN    = /usr/local/include/eigen3/

INCLUDE = -I$(EIGEN)

.PHONY: all clean cleanest
all: np_divs

OBJS = div_func.o div_l2.o div_alpha.o div_renyi.o div_bc.o div_hellinger.o \
	   np_divs.o gamma.o utils.o

################################################################################
### General rule for making .o files that respects .h dependencies
### (based on http://scottmcpeak.com/autodepend/autodepend.html)

# pull in dependency info for preexisting .o files
-include $(OBJS:.o=.d)

# general rule for compiling and making dependency info
# creates command-less, prereq-less targets to avoid errors when renaming files
%.o: %.cpp
	@# actual compilation
	$(CPP) -c $(CPPFLAGS) $(INCLUDE) $*.cpp -o $*.o
	@
	@# cache mangled dependency info in filename.d
	@$(CPP) -MM -MP $(CPPFLAGS) $(INCLUDE) $*.cpp > $*.d 


################################################################################
### Divergence estimator

np_divs: $(OBJS)
	$(CPP) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)


################################################################################
### Cleanup
clean:
	rm -f np_divs *.o *.d
