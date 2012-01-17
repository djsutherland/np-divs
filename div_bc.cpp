#include "div_bc.hpp"

DivBC* DivBC::do_clone() const {
    return new DivBC(ub);
}
