CPP      = clang++
CPPFLAGS = -Wall -g
LDFLAGS  = -lflann
EIGEN    = /usr/local/include/eigen3/

INCLUDE = -I$(EIGEN)

.PHONY: all clean cleanest
all: np_divs

OBJS = div_func.o div_l2.o div_alpha.o div_renyi.o div_bc.o div_hellinger.o \
	   np_divs.o dkn.o gamma.o utils.o

################################################################################
### General rule for making .o files that respects .h dependencies
### (based on http://www.makelinux.net/make3/make3-CHP-8-SECT-3)

# pull in dependency info for preexisting .o files
-include $(OBJS:.o=.d)

# $(call make-depend, source-file, object-file, depend-file)
define make-depend
  $(CPP) -MM -MP -MT $2 $(CPPFLAGS) $(INCLUDE) $1 > $3
endef

# general rule for compiling and making dependency info
%.o: %.cpp
	$(call make-depend,$<,$@,$(subst .o,.d,$@))
	$(CPP) -c $(CPPFLAGS) $(INCLUDE) $<


################################################################################
### Divergence estimator

np_divs: $(OBJS)
	$(CPP) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)


################################################################################
### Cleanup
clean:
	rm -f np_divs *.o *.d
