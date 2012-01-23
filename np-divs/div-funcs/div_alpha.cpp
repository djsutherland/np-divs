#include "np-divs/div-funcs/div_alpha.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <boost/bind.hpp>

#include "np-divs/fix_terms.hpp"
#include "np-divs/gamma.hpp"

namespace NPDivs {

using namespace std;

DivAlpha::DivAlpha(double alpha_, double ub_) : DivFunc(ub_), alpha(alpha_) {}

string DivAlpha::name() const {
    return (boost::format("Alpha %g divergence") % alpha).str();
}


double DivAlpha::operator()(const vector<float> &rho_x,
                            const vector<float> &nu_x,
                            const vector<float> &rho_y,
                            const vector<float> &nu_y,
                            int dim,
                            int k) const {
    /* Estimates alpha-divergence \int p^\alpha q^(1-\alpha) based on
     * kth-nearest-neighbor statistics.
     *
     * Note that rho_y is used only for its .size(), and nu_y is not used at
     * all. (They're there to be consistent with the DivFunc interface.)
     */

    return (*this)(rho_x, nu_x, rho_y.size(), dim, k);
}

double DivAlpha::operator()(const vector<float> &rho,
                            const vector<float> &nu,
                            int m,
                            int dim,
                            int k) const {
    /* Estimates alpha-divergence \int p^\alpha q^(1-\alpha) based on
     * kth-nearest-neighbor statistics.
     *
     * m is the number of sample points in the Y distribution (the one
     * that nu is computed relative to).
     */
    using boost::bind;

    size_t n = rho.size();

    // r = rho_x ./ nu_x
    vector<float> r;
    r.resize(n);
    transform(rho.begin(), rho.end(), nu.begin(), r.begin(),
            divides<float>());
    
    // cap anything too big
    fix_terms(r);

    // r = r .^ (dim * (1-alpha))
    transform(r.begin(), r.end(), r.begin(),
            bind<double, double(&)(double,double)>(pow, _1, dim*(1-alpha)));

    // find the mean of r and multiply by the appropriate constant
    return accumulate(r.begin(), r.end(), 0.) / n *
           exp(lgamma(k)*2 - lgamma(k+1-alpha) - lgamma(k+alpha-1)) *
           pow((n-1.0) / m, 1.-alpha);
}

DivAlpha* DivAlpha::do_clone() const {
    return new DivAlpha(alpha, ub);
}

}
