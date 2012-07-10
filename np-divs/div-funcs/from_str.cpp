/*******************************************************************************
 * Copyright (c) 2012, Dougal J. Sutherland (dsutherl@cs.cmu.edu).             *
 * All rights reserved.                                                        *
 *                                                                             *
 * Redistribution and use in source and binary forms, with or without          *
 * modification, are permitted provided that the following conditions are met: *
 *                                                                             *
 *     * Redistributions of source code must retain the above copyright        *
 *       notice, this list of conditions and the following disclaimer.         *
 *                                                                             *
 *     * Redistributions in binary form must reproduce the above copyright     *
 *       notice, this list of conditions and the following disclaimer in the   *
 *       documentation and/or other materials provided with the distribution.  *
 *                                                                             *
 *     * Neither the name of Carnegie Mellon University nor the                *
 *       names of the contributors may be used to endorse or promote products  *
 *       derived from this software without specific prior written permission. *
 *                                                                             *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" *
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE   *
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE  *
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE   *
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR         *
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF        *
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS    *
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN     *
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)     *
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE  *
 * POSSIBILITY OF SUCH DAMAGE.                                                 *
 ******************************************************************************/
#include "np-divs/div-funcs/from_str.hpp"

#include "np-divs/div-funcs/div_func.hpp"
#include "np-divs/div-funcs/div_alpha.hpp"
#include "np-divs/div-funcs/div_bc.hpp"
#include "np-divs/div-funcs/div_hellinger.hpp"
#include "np-divs/div-funcs/div_l2.hpp"
#include "np-divs/div-funcs/div_linear.hpp"
#include "np-divs/div-funcs/div_renyi.hpp"

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/algorithm/string/split.hpp>
#include <boost/bind.hpp>
#include <boost/throw_exception.hpp>

using std::bind2nd;
using std::equal_to;
using std::string;
using std::vector;

using boost::algorithm::split;

namespace npdivs {

#define THROW_DOM(x)\
    BOOST_THROW_EXCEPTION(std::domain_error(x))

DivFunc* div_func_from_str(const string &spec) {
    vector<string> tokens;
    split(tokens, spec, bind2nd(equal_to<char>(), ':'));

    size_t num_toks = tokens.size();
    if (num_toks == 0)
        THROW_DOM("can't handle empty div func specification");

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
            default: THROW_DOM("too many arguments for DivAlpha");
        }

    } else if (kind == "bc") {
        switch (num_toks) {
            case 2: return new DivBC(args[0]);
            case 1: return new DivBC();
            default: THROW_DOM("too many arguments for DivBC");
        }

    } else if (kind == "hellinger") {
        switch (num_toks) {
            case 2: return new DivHellinger(args[0]);
            case 1: return new DivHellinger();
            default: THROW_DOM("too many arguments for DivHellinger");
        }

    } else if (kind == "l2") {
        switch (num_toks) {
            case 2: return new DivL2(args[0]);
            case 1: return new DivL2();
            default: THROW_DOM("too many arguments for DivL2");
        }

    } else if (kind == "linear") {
        switch (num_toks) {
            case 2: return new DivLinear(args[0]);
            case 1: return new DivLinear();
            default: THROW_DOM("too many arguments for DivLinear");
        }

    } else if (kind == "renyi") {
        switch (num_toks) {
            case 3: return new DivRenyi(args[0], args[1]);
            case 2: return new DivRenyi(args[0]);
            case 1: return new DivRenyi();
            default: THROW_DOM("too many arguments for DivRenyi");
        }

    } else {
        THROW_DOM("unknown div func type '" + kind + "'");
    }
}

}
