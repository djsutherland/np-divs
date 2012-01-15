#define _USE_MATH_DEFINES // to get M_PI
#include <cmath>

#include <vector>

#include "div_l2.hpp"
#include "fix_terms.hpp"
#include "gamma_half.hpp"

using Eigen::VectorXf;
using Eigen::ArrayXf;

DivL2::DivL2(unsigned int dim_, unsigned int k_, double ub_)
    : DivFunc(dim_, k_, ub_) {}

// estimates L2 divergence between X and Y: \sqrt \int (p-q)^2
double DivL2::operator()(const VectorXf &rho_x,
                         const VectorXf &nu_x,
                         const VectorXf &rho_y,
                         const VectorXf &nu_y) const {
    const double c = pow(M_PI, .5 * dim) / gamma_half(dim + 2);
    const double scale = (k-1) / c;

    int M = rho_x.size();
    int N = rho_y.size();

    // break up the calculation according to
    // \sqrt \int (p - q)^2 = \sqrt( \int p^2 - \int pq - \int qp + \int q^2 )
    const ArrayXf t1x = rho_x.array().pow(-dim) / (N-1); // \int p^2
    const ArrayXf t3x =  nu_x.array().pow(-dim) / M;     // \int pq
    const ArrayXf t3y =  nu_y.array().pow(-dim) / N;     // \int qp
    const ArrayXf t1y = rho_y.array().pow(-dim) / (M-1); // \int q^2

    // combine
    ArrayXf tterm = (t1x - t3x - t3y + t1y) * scale;

    // throw away anything that's inf or otherwise too large
    std::vector<float> vec = as_vector(tterm);
    fix_terms(vec, ub);
    Eigen::Map<VectorXf> final(&vec[0], vec.size());

    // take the mean
    double res = final.sum() / final.size();
    return res > 0 ? sqrt(res) : 0;
}
