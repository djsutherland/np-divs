#include "div_hellinger.hpp"

#include <cmath>
#include <string>

namespace NPDivs {

std::string DivHellinger::name() const {
    return "Hellinger distance";
}

double DivHellinger::operator()(const std::vector<float> &rho,
                                const std::vector<float> &nu,
                                int m,
                                int dim,
                                int k) const {
    double est = this->super::operator()(rho, nu, m, dim, k);
    return est < 1 ? std::sqrt(1 - est) : 0;
}

DivHellinger* DivHellinger::do_clone() const {
    return new DivHellinger(ub);
}

}
