#ifndef DIV_BC_HPP
#define DIV_BC_HPP
#include "np-divs/basics.hpp"

#include <string>

#include "np-divs/div-funcs/div_alpha.hpp"

namespace NPDivs {

class DivBC : public DivAlpha {
    typedef DivAlpha super;

    public:
        DivBC(double ub=.99) : super(.5, ub) {};

        virtual std::string name() const;

    private:
        DivBC* do_clone() const;
};

}

#endif
