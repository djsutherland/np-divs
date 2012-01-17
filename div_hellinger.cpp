#include "div_hellinger.hpp"

#include <cmath>

double DivHellinger::operator()(const std::vector<float> &rho_x,
                                const std::vector<float> &nu_x,
                                const std::vector<float> &rho_y,
                                const std::vector<float> &nu_y,
                                unsigned int dim,
                                unsigned int k) const {
    double est = this->super::operator()(rho_x, nu_x, rho_y, nu_y, dim, k);
    return est < 1 ? std::sqrt(1 - est) : 0;
}

DivHellinger* DivHellinger::do_clone() const {
    return new DivHellinger(ub);
}
