#ifndef DIV_FUNC_HPP_
#define DIV_FUNC_HPP_

#include <Eigen/Core>

class DivFunc {
    protected:
        double ub; // if ub is .99, will cap terms at the 99-th percentile

    public:
        DivFunc(double ub = .99);

        virtual double operator()(
                Eigen::VectorXf rho_x,
                Eigen::VectorXf nu_x,
                Eigen::VectorXf rho_y,
                Eigen::VectorXf nu_y,
                unsigned int dim,
                unsigned int k
            ) const = 0;
};
#endif
