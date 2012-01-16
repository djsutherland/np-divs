#include <cmath>

#include "div_hellinger.hpp"

double DivHellinger::operator()(const Eigen::VectorXf &rho_x,
                                const Eigen::VectorXf &nu_x,
                                const Eigen::VectorXf &rho_y,
                                const Eigen::VectorXf &nu_y,
                                unsigned int dim,
                                unsigned int k) const {
    double est = this->super::operator()(rho_x, nu_x, rho_y, nu_y, dim, k);
    return est < 1 ? std::sqrt(1 - est) : 0;
}
