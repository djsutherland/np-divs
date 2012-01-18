#ifndef DIV_RENYI_HPP
#define DIV_RENYI_HPP
#include "basics.hpp"

#include <vector>

#include "div_alpha.hpp"

class DivRenyi : public DivAlpha {
    typedef DivAlpha super;

    public:
        DivRenyi(double alpha=.999, double ub=.99);

        virtual double operator()(
                const std::vector<float> &rho_x,
                const std::vector<float> &nu_x,
                int m,
                int dim,
                int k
            ) const;

    private:
        virtual DivRenyi* do_clone() const;
};

#endif
