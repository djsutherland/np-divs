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
#include "np-divs/div-funcs/div_l2.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/throw_exception.hpp>

#include "np-divs/fix_terms.hpp"
#include "np-divs/gamma.hpp"

namespace npdivs {

using namespace std;

DivL2::DivL2(double ub_) : DivFunc(ub_) {}

string DivL2::name() const {
    return "L2 divergence";
}


class pow_mult {
    double ex;
    double mult;
public:
    pow_mult(double e, double m) : ex(e), mult(m) {}
    double operator()(double x) {
        return pow(x, ex) * mult;
    }
};

template <typename T>
inline T mean(const vector<T> &v) {
    return accumulate(v.begin(), v.end(), (T) 0) / v.size();
}


double DivL2::operator()(const vector<float> &rho_x,
                         const vector<float> &nu_x,
                         const vector<float> &rho_y,
                         const vector<float> &nu_y,
                         int dim,
                         int k) const {
    /* Estimates L2 divergence \sqrt \int (p-q)^2 between distribution X and Y,
     * based on kth-nearest-neighbor statistics.
     */
    if (k <= 1) {
        BOOST_THROW_EXCEPTION(domain_error(
                    "l2 divergence estimator needs k >= 2"));
    }
    // (k-1) / volume of unit ball: this is B_{k,a,b} for a=0,b=1 and a=1,b=0
    const double c = (k-1) / pow(M_PI, .5 * dim) * gamma(dim/2.0 + 1);
    // XXX only works for dimensions up to 340

    int N = rho_x.size();
    int M = rho_y.size();

    // break up the calculation according to
    // \sqrt \int (p - q)^2 = \sqrt( \int p^2 - \int qp - \int pq + \int q^2 )
    vector<double> pp, qp, pq, qq;
    pp.resize(N); qp.resize(N);
    pq.resize(M); qq.resize(M);

    transform(rho_x.begin(), rho_x.end(), pp.begin(), pow_mult(-dim, c/(N-1)));
    transform( nu_x.begin(),  nu_x.end(), qp.begin(), pow_mult(-dim, c/  M  ));
    transform( nu_y.begin(),  nu_y.end(), pq.begin(), pow_mult(-dim, c/  N  ));
    transform(rho_y.begin(), rho_y.end(), qq.begin(), pow_mult(-dim, c/(M-1)));

    double res;
    if (N != M) {
        // throw away anything too big
        fix_terms(pp, ub);
        fix_terms(qp, ub);
        fix_terms(pq, ub);
        fix_terms(qq, ub);

        // combine terms
        res = mean(pp) - mean(qp) - mean(pq) + mean(qq);

    } else {
        // this is slightly faster, and more consistent with the matlab code
        // TODO - this special case should probably go away eventually
        for (int i = 0; i < N; i++) {
            pp[i] += qq[i] - pq[i] - qp[i];
        }

        fix_terms(pp, ub);
        res = mean(pp);
    };
    return res > 0 ? sqrt(res) : 0.;
}

DivL2* DivL2::do_clone() const {
    return new DivL2(ub);
}

}
