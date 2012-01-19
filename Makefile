CPP      = clang++
CPPFLAGS = -Wall -g -O0

LDFLAGS  = -lflann -lhdf5 -lboost_thread-mt
GTEST    = -lgtest

INCLUDE =

.PHONY: all clean debug test

all: tests
debug: tests.dSYM
	gdb ./tests

OBJS = div_func.o div_l2.o div_alpha.o div_renyi.o div_bc.o div_hellinger.o \
	   dkn.o gamma.o fix_terms.o np_divs.o
ALLOBJS = $(OBJS) tests.o

################################################################################
### General rule for making .o files that respects .h dependencies
### (based on http://www.makelinux.net/make3/make3-CHP-8-SECT-3)

# pull in dependency info for preexisting .o files
-include $(ALLOBJS:.o=.d)

# $(call make-depend, source-file, object-file, depend-file)
define make-depend
  $(CPP) -MM -MP -MT $2 $(CPPFLAGS) $(INCLUDE) $1 > $3
endef

# general rule for compiling and making dependency info
%.o: %.cpp
	@$(call make-depend,$<,$@,$(subst .o,.d,$@))
	$(CPP) -c $(CPPFLAGS) $(INCLUDE) $<


################################################################################
### Binaries

tests: tests.o $(OBJS)
	$(CPP) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) $(GTEST)

%.dSYM: %
	dsymutil $<

################################################################################
### Other

test: tests
	./tests

clean:
	rm -rf np_divs tests *.o *.d *.dSYM
