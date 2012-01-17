#include "div_alpha.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <boost/bind.hpp>

#include "fix_terms.hpp"
#include "gamma.hpp"

using namespace std;

DivAlpha::DivAlpha(double alpha_, double ub_) : DivFunc(ub_) {
    if (alpha_ == 1.0) {
        throw std::domain_error("alpha of 1.0 is not useful");
    }
    alpha = alpha_;
}

double DivAlpha::operator()(const vector<float> &rho_x,
                            const vector<float> &nu_x,
                            unsigned int dim,
                            unsigned int k) const {
    vector<float> fake;
    return (*this)(rho_x, nu_x, fake, fake, dim, k);
}

double DivAlpha::operator()(const vector<float> &rho_x,
                            const vector<float> &nu_x,
                            const vector<float> &rho_y,
                            const vector<float> &nu_y,
                            unsigned int dim,
                            unsigned int k) const {
    /* Estimates alpha-divergence \int p^\alpha q^(1-\alpha) based on
     * kth-nearest-neighbor statistics.
     *
     * Note that rho_y and nu_y are unused and may be empty. (They're there
     * to be consistent with the DivFunc interface.)
     */

    using namespace boost;

    size_t n = rho_x.size();

    vector<float> r;
    r.reserve(n);

    // r = rho_x ./ nu_x
    transform(rho_x.begin(), rho_x.end(), nu_x.begin(), r.begin(),
            divides<float>());
    
    // cap anything too big
    fix_terms(r);

    // r = r .^ (dim * (1-alpha))
    transform(r.begin(), r.end(), r.begin(),
            bind<double, double(&)(double,double)>(pow, _1, dim*(1-alpha)));

    // find the mean of r and multiply by the appropriate constant
    return accumulate(r.begin(), r.end(), 0) / n *
           exp(lgamma(k)*2 - lgamma(k+1-alpha) - lgamma(k+alpha-1)) *
           pow((rho_x.size()-1.0)/nu_x.size(), 1-alpha);
}

DivAlpha* DivAlpha::do_clone() const {
    return new DivAlpha(alpha, ub);
}
