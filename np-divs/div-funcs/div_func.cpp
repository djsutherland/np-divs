#include "np-divs/div-funcs/div_func.hpp"

namespace NPDivs {

DivFunc::DivFunc(double ub_) {
    ub = ub_;
}

DivFunc* DivFunc::clone() const {
    return do_clone();
}

}
