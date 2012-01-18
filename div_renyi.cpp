#include "div_renyi.hpp"

#include <cmath>
#include <vector>

DivRenyi::DivRenyi(double alpha, double ub) : super(alpha, ub) {}

double DivRenyi::operator()(const std::vector<float> &rho_x,
                            const std::vector<float> &nu_x,
                            const std::vector<float> &rho_y,
                            const std::vector<float> &nu_y,
                            int dim,
                            int k) const {
    /* Estimates Renyi divergence \log (\int p^\alpha q^(1-\alpha)) / (\alpha-1)
     * based on kth-nearest-neighbor statistics.
     */

    double est = this->super::operator()(rho_x, nu_x, rho_y, nu_y, dim, k);
    return est > 1 ? std::log(est) / (alpha - 1.0) : 0;
}

DivRenyi* DivRenyi::do_clone() const {
    return new DivRenyi(alpha, ub);
}
