#include <cmath>

#include "div_renyi.hpp"

using Eigen::VectorXf;

double DivRenyi::operator()(const VectorXf &rho_x,
                            const VectorXf &nu_x,
                            const VectorXf &rho_y,
                            const VectorXf &nu_y,
                            unsigned int dim,
                            unsigned int k) const {
    /* Estimates Renyi divergence \log (\int p^\alpha q^(1-\alpha)) / (\alpha-1)
     * based on kth-nearest-neighbor statistics.
     */

    double est = this->super::operator()(rho_x, nu_x, rho_y, nu_y, dim, k);
    return est > 1 ? std::log(est) / (alpha - 1.0) : 0;
}
