#include "div_l2.hpp"

#include <cmath>
#include <vector>
#include <Eigen/Core>

#include "gamma.hpp"
#include "utils.hpp"

using Eigen::VectorXf;
using Eigen::ArrayXf;

DivL2::DivL2(double ub_) : DivFunc(ub_) {}

double DivL2::operator()(const VectorXf &rho_x,
                         const VectorXf &nu_x,
                         const VectorXf &rho_y,
                         const VectorXf &nu_y,
                         unsigned int dim,
                         unsigned int k) const {
    /* Estimates L2 divergence \sqrt \int (p-q)^2 between distribution X and Y,
     * based on kth-nearest-neighbor statistics.
     */
    const double c = pow(M_PI, .5 * dim) / gamma(dim/2.0 + 1);
    const double con = (k-1) / c;

    int M = rho_x.size();
    int N = rho_y.size();

    // break up the calculation according to
    // \sqrt \int (p - q)^2 = \sqrt( \int p^2 - \int pq - \int qp + \int q^2 )
    const ArrayXf t1x = rho_x.array().pow(-dim) / (N-1); // \int p^2
    const ArrayXf t3x =  nu_x.array().pow(-dim) / M;     // \int pq
    const ArrayXf t3y =  nu_y.array().pow(-dim) / N;     // \int qp
    const ArrayXf t1y = rho_y.array().pow(-dim) / (M-1); // \int q^2

    // combine and throw away anything that's inf or otherwise too large
    Eigen::Map<ArrayXf> final = fixed_terms((t1x - t3x - t3y + t1y) * con, ub);

    // take the mean and sqrt it
    double res = final.sum() / final.size();
    return res > 0 ? sqrt(res) : 0;
}
