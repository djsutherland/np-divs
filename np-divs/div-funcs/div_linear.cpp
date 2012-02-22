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
#include "np-divs/div-funcs/div_linear.hpp"

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

DivLinear::DivLinear(double ub_) : DivFunc(ub_) {}

string DivLinear::name() const {
    return "Linear divergence";
}


double DivLinear::operator()(const vector<float> &rho_x,
                             const vector<float> &nu_x,
                             const vector<float> &rho_y,
                             const vector<float> &nu_y,
                             int dim,
                             int k) const {
    /* Estimates linear "divergence" \int qp based on kth-nearest-neighbor
     * statistics.
     *
     * Note that rho_y is used only for its .size(), and nu_y is not used at
     * all. (They're there to be consistent with the DivFunc interface.)
     */

    return (*this)(rho_x, nu_x, rho_y.size(), dim, k);
}

double DivLinear::operator()(const vector<float> &rho,
                            const vector<float> &nu,
                            int m,
                            int dim,
                            int k) const {
    /* Estimates linear "divergence" \int qp based on kth-nearest-neighbor
     * statistics.
     *
     * m is the number of sample points in the Y distribution (the one
     * that nu is computed relative to).
     */
    size_t n = rho.size();

    // r = nu ^ -d
    vector<float> r;
    r.resize(n);
    transform(nu.begin(), nu.end(), r.begin(),
            boost::bind<double, double(&)(double,double)>(pow, _1, -1.*dim));

    // cap anything too big
    fix_terms(r, ub);

    // find the mean of r and multiply by the appropriate constant
    return accumulate(r.begin(), r.end(), 0.0) / ((double) n)
           * (k-1.) // gamma(k)^2 / gamma(k-0) / gamma(k-1)
           / pow(M_PI, .5 * dim) * gamma(dim/2.0 + 1) // vol. of unit ball
           / ((double) m);
}

DivLinear* DivLinear::do_clone() const {
    return new DivLinear(ub);
}

}
