#include "div_l2.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include "fix_terms.hpp"
#include "gamma.hpp"

namespace NPDivs {

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
    const double c = (k-1) / pow(M_PI, .5 * dim) * gamma(dim/2.0 + 1);

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
        fix_terms(pp);
        fix_terms(qp);
        fix_terms(pq);
        fix_terms(qq);

        // combine terms
        res = mean(pp) - mean(qp) - mean(pq) + mean(qq);

    } else {
        // this is slightly faster, and more consistent with the matlab code
        // TODO - this special case should probably go away eventually
        for (size_t i = 0; i < N; i++) {
            pp[i] += qq[i] - pq[i] - qp[i];
        }

        fix_terms(pp);
        res = mean(pp);
    };
    return res > 0 ? sqrt(res) : 0.;
}

DivL2* DivL2::do_clone() const {
    return new DivL2(ub);
}

}
