#ifndef DIV_BC_HPP
#define DIV_BC_HPP
#include "basics.hpp"

#include "div_alpha.hpp"

class DivBC : public DivAlpha {
    typedef DivAlpha super;

    public:
        DivBC(double ub=.99) : super(.5, ub) {};

    private:
        DivBC* do_clone() const;
};

#endif
