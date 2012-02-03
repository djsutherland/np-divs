#ifndef DIV_FUNC_FROM_STR_HPP_
#define DIV_FUNC_FROM_STR_HPP_
#include "np-divs/basics.hpp"
#include "np-divs/div-funcs/div_func.hpp"

#include <string>

namespace NPDivs{

DivFunc* div_func_from_str(std::string spec);

}
#endif
