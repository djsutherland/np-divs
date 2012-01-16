#ifndef DIV_ALPHA_HPP_
#define DIV_ALPHA_HPP_

#include <Eigen/Core>

#include "div_func.hpp"

class DivAlpha : public DivFunc {
    typedef DivFunc super;

    protected:
        double alpha;

    public:
        DivAlpha(double alpha=.999, double ub = .99);

        virtual double operator()(
                const Eigen::VectorXf &rho_x,
                const Eigen::VectorXf &nu_x,
                unsigned int dim,
                unsigned int k
            ) const;

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
