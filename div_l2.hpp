#ifndef DIV_L2_HPP_
#define DIV_L2_HPP_

#include <Eigen/Core>

#include "div_func.hpp"

class DivL2 : public DivFunc {
    public:
        DivL2(double ub = .99);

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
