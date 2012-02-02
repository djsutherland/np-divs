#ifndef DIV_HELLINGER_HPP
#define DIV_HELLINGER_HPP
#include "np-divs/basics.hpp"

#include <string>
#include <vector>

#include "np-divs/div-funcs/div_alpha.hpp"

namespace NPDivs {

class DivHellinger : public DivAlpha {
    typedef DivAlpha super;

    public:
        DivHellinger(double ub=.99) : super(.5, ub) {};

        virtual std::string name() const;

        virtual double operator()(
                const std::vector<float> &rho,
                const std::vector<float> &nu,
                int m,
                int dim,
                int k
            ) const;

    private:
        DivHellinger* do_clone() const;
};

}

#endif