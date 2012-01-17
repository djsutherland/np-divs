#include "div_func.hpp"

DivFunc::DivFunc(double ub_) {
    ub = ub_;
}

DivFunc* DivFunc::clone() const {
    return do_clone();
}
