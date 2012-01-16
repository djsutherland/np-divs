#ifndef DIV_BC_HPP
#define DIV_BC_HPP

#include <Eigen/Core>
#include "div_alpha.hpp"

class DivBC : public DivAlpha {
    typedef DivAlpha super;

    public:
        DivBC(double ub=.99) : super(.5, ub) {};
};

#endif
