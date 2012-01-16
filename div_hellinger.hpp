#ifndef DIV_HELLINGER_HPP
#define DIV_HELLINGER_HPP
#include "basics.hpp"

#include <Eigen/Core>
#include "div_alpha.hpp"

class DivHellinger : public DivAlpha {
    typedef DivAlpha super;

    public:
        DivHellinger(double ub=.99) : super(.5, ub) {};

        virtual double operator()(
                const Eigen::VectorXf &rho_x,
                const Eigen::VectorXf &nu_x,
                const Eigen::VectorXf &rho_y,
                const Eigen::VectorXf &nu_y,
                unsigned int dim,
                unsigned int k
            ) const;
};

#endif
