#ifndef DIV_RENYI_HPP
#define DIV_RENYI_HPP

#include <Eigen/Core>
#include "div_alpha.hpp"

class DivRenyi : public DivAlpha {
    typedef DivAlpha super;

    public:
        DivRenyi(double alpha=.999, double ub=.99);

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
