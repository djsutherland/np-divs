#include "div_alpha.hpp"

#include <cmath>
#include <stdexcept>
#include <Eigen/Core>

#include "gamma.hpp"
#include "utils.hpp"

using namespace Eigen;
using namespace std;

DivAlpha::DivAlpha(double alpha_, double ub_) : DivFunc(ub_) {
    if (alpha_ == 1.0) {
        throw std::domain_error("alpha of 1.0 is not useful");
    }
    alpha = alpha_;
}

double DivAlpha::operator()(const VectorXf &rho_x,
                            const VectorXf &nu_x,
                            unsigned int dim,
                            unsigned int k) const {
    VectorXf fake;
    return (*this)(rho_x, nu_x, fake, fake, dim, k);
}

double DivAlpha::operator()(const VectorXf &rho_x,
                            const VectorXf &nu_x,
                            const VectorXf &rho_y,
                            const VectorXf &nu_y,
                            unsigned int dim,
                            unsigned int k) const {
    /* Estimates alpha-divergence \int p^\alpha q^(1-\alpha) based on
     * kth-nearest-neighbor statistics.
     *
     * Note that rho_y and nu_y are unused and may be empty. (They're there
     * to be consistent with the DivFunc interface.)
     */

    const ArrayXf r = rho_x.array() / nu_x.array();
    Eigen::Map<ArrayXf> final = fixed_terms(r, ub);
    final = final.pow(dim * (1-alpha));

    return final.sum() / final.size() *
           exp(lgamma(k)*2 - lgamma(k+1-alpha) - lgamma(k+alpha-1)) *
           pow((rho_x.size()-1.0)/nu_x.size(), 1-alpha);
}
