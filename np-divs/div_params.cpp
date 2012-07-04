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
#include "np-divs/div_params.hpp"

#include <stdexcept>
#include <boost/throw_exception.hpp>

#include <flann/flann.hpp>

namespace npdivs {

void do_nothing(size_t left) {}
void print_progress_cerr(size_t left) {
    std::cerr << left << " pairs left to compute\n";
}

// TODO: more index types, support arguments
flann::IndexParams index_params_from_str(const std::string &spec) {
    // even though this looks like object slicing, it's not, i promise
    if (spec == "linear" || spec == "brute") {
        flann::LinearIndexParams ps;
        return ps;
    } else if (spec == "kdtree" || spec == "kd") {
        flann::KDTreeSingleIndexParams ps;
        return ps;
    } else {
        BOOST_THROW_EXCEPTION(std::domain_error("unknown index type " + spec));
    }
}

}
