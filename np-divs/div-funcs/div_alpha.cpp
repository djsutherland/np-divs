/*******************************************************************************
 * Copyright (c) 2012, Dougal J. Sutherland (dsutherl@cs.cmu.edu).             *
 * All rights reserved.                                                        *
 *                                                                             *
 * Redistribution and use in source and binary forms, with or without          *
 * modification, are permitted provided that the following conditions are met: *
 *                                                                             *
 *     * Redistributions of source code must retain the above copyright        *
 *       notice, this list of conditions and the following disclaimer.         *
 *                                                                             *
 *     * Redistributions in binary form must reproduce the above copyright     *
 *       notice, this list of conditions and the following disclaimer in the   *
 *       documentation and/or other materials provided with the distribution.  *
 *                                                                             *
 *     * Neither the name of Carnegie Mellon University nor the                *
 *       names of the contributors may be used to endorse or promote products  *
 *       derived from this software without specific prior written permission. *
 *                                                                             *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" *
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE   *
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE  *
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE   *
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR         *
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF        *
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS    *
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN     *
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)     *
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE  *
 * POSSIBILITY OF SUCH DAMAGE.                                                 *
 ******************************************************************************/
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

namespace npdivs {

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
    fix_terms(r, ub);

    // r = r .^ (dim * (1-alpha))
    transform(r.begin(), r.end(), r.begin(),
            bind<double, double(&)(double,double)>(pow, _1, dim*(1-alpha)));

    // find the mean of r and multiply by the appropriate constant
    return accumulate(r.begin(), r.end(), 0.) / n *
           exp(lgamma(k)*2 - lgamma(k+1-alpha) - lgamma(k+alpha-1)) *
           pow((n-1.0) / m, 1.-alpha);
    // FIXME: what about the c-bar term?
}

double DivAlpha::get_alpha() const { return alpha; }

DivAlpha* DivAlpha::do_clone() const {
    return new DivAlpha(alpha, ub);
}

}
