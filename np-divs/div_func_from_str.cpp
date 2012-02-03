#include "div_func_from_str.hpp"

#include "np-divs/div-funcs/div_func.hpp"
#include "np-divs/div-funcs/div_alpha.hpp"
#include "np-divs/div-funcs/div_bc.hpp"
#include "np-divs/div-funcs/div_hellinger.hpp"
#include "np-divs/div-funcs/div_l2.hpp"
#include "np-divs/div-funcs/div_renyi.hpp"

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/algorithm/string/split.hpp>
#include <boost/bind.hpp>
#include <boost/format.hpp>

using std::bind2nd;
using std::domain_error;
using std::equal_to;
using std::string;
using std::vector;

using boost::algorithm::split;
using boost::format;

namespace NPDivs {

DivFunc* div_func_from_str(string spec) {
    vector<string> tokens;
    split(tokens, spec, bind2nd(equal_to<char>(), ':'));

    size_t num_toks = tokens.size();
    if (num_toks == 0)
        throw domain_error("can't handle empty div func specification");

    const string &kind = tokens[0];

    vector<double> args;
    args.reserve(num_toks - 1);
    for (size_t i = 1; i < num_toks; i++)
        args.push_back(atof(tokens[i].c_str()));

    if (kind == "alpha") {
        switch (num_toks) {
            case 3: return new DivAlpha(args[0], args[1]);
            case 2: return new DivAlpha(args[0]);
            case 1: return new DivAlpha();
            default: throw domain_error("too many arguments for DivAlpha");
        }

    } else if (kind == "bc") {
        switch (num_toks) {
            case 2: return new DivBC(args[0]);
            case 1: return new DivBC();
            default: throw domain_error("too many arguments for DivBC");
        }

    } else if (kind == "hellinger") {
        switch (num_toks) {
            case 2: return new DivHellinger(args[0]);
            case 1: return new DivHellinger();
            default: throw domain_error("too many arguments for DivHellinger");
        }

    } else if (kind == "l2") {
        switch (num_toks) {
            case 2: return new DivL2(args[0]);
            case 1: return new DivL2();
            default: throw domain_error("too many arguments for DivL2");
        }

    } else if (kind == "renyi") {
        switch (num_toks) {
            case 3: return new DivRenyi(args[0], args[1]);
            case 2: return new DivRenyi(args[0]);
            case 1: return new DivRenyi();
            default: throw domain_error("too many arguments for DivRenyi");
        }

    } else {
        throw domain_error((format("unknown div func type %d") % kind).str());
    }
}

}
