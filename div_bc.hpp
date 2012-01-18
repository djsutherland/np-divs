#ifndef DIV_BC_HPP
#define DIV_BC_HPP
#include "basics.hpp"

#include <string>

#include "div_alpha.hpp"

class DivBC : public DivAlpha {
    typedef DivAlpha super;

    public:
        DivBC(double ub=.99) : super(.5, ub) {};

        virtual std::string name() const;

    private:
        DivBC* do_clone() const;
};

#endif
