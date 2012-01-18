#include "div_renyi.hpp"

#include <boost/format.hpp>
#include <cmath>
#include <vector>

DivRenyi::DivRenyi(double alpha, double ub) : super(alpha, ub) {}

std::string DivRenyi::name() const {
    return (boost::format("Renyi-%g divergence") % alpha).str();
}

double DivRenyi::operator()(const std::vector<float> &rho,
                            const std::vector<float> &nu,
                            int m,
                            int dim,
                            int k) const {
    /* Estimates Renyi divergence \log (\int p^\alpha q^(1-\alpha)) / (\alpha-1)
     * based on kth-nearest-neighbor statistics.
     */

    double est = this->super::operator()(rho, nu, m, dim, k);
    return std::max(0., std::log(est) / (alpha - 1.));
}

DivRenyi* DivRenyi::do_clone() const {
    return new DivRenyi(alpha, ub);
}
