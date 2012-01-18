#include "div_bc.hpp"

std::string DivBC::name() const {
    return "Bhattacharyya coefficient";
}

DivBC* DivBC::do_clone() const {
    return new DivBC(ub);
}
