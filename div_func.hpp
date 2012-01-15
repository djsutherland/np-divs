#ifndef DIV_FUNC_HPP_
#define DIV_FUNC_HPP_

#include <Eigen/Core>

class DivFunc {
    protected:
        unsigned int dim; // dimensionality of data
        unsigned int k; // how many nearest neighbors being used
        double ub; // if ub is .99, will cap terms at the 99-th percentile

    public:
        DivFunc(unsigned int dim, unsigned int k, double ub = .99);

        virtual double operator()(
                Eigen::VectorXf rho_x,
                Eigen::VectorXf nu_x,
                Eigen::VectorXf rho_y,
                Eigen::VectorXf nu_y
            ) const = 0;
};
#endif
