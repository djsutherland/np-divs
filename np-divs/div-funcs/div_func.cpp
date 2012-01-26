#include "np-divs/div-funcs/div_func.hpp"

namespace NPDivs {

DivFunc::DivFunc(double ub_) : ub(ub_) { }

double DivFunc::get_ub() const { return ub; }

DivFunc* DivFunc::clone() const { return do_clone(); }

}
