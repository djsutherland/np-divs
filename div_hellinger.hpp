#ifndef DIV_HELLINGER_HPP
#define DIV_HELLINGER_HPP
#include "basics.hpp"

#include <vector>

#include "div_alpha.hpp"

class DivHellinger : public DivAlpha {
    typedef DivAlpha super;

    public:
        DivHellinger(double ub=.99) : super(.5, ub) {};

        virtual double operator()(
                const std::vector<float> &rho_x,
                const std::vector<float> &nu_x,
                const std::vector<float> &rho_y,
                const std::vector<float> &nu_y,
                int dim,
                int k
            ) const;

    private:
        DivHellinger* do_clone() const;
};

#endif
