CPP      = clang++
CPPFLAGS = -Wall -g -O0

LDFLAGS  = -lflann -lhdf5
GTEST    = -lgtest

INCLUDE =

.PHONY: all clean debug
all: np_divs
debug: np_divs.dSYM tests

OBJS = div_func.o div_l2.o div_alpha.o div_renyi.o div_bc.o div_hellinger.o \
	   dkn.o gamma.o fix_terms.o
ALLOBJS = $(OBJS) np_divs.o tests.o

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

np_divs: np_divs.o $(OBJS)
	$(CPP) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

tests: tests.o $(OBJS)
	$(CPP) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) $(GTEST)

np_divs.dSYM: np_divs
	dsymutil np_divs


################################################################################
### Cleanup
clean:
	rm -f np_divs tests *.o *.d
