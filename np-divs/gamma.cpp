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
#include "np-divs/gamma.hpp"

// based on code from http://www.crbond.com/math.htm:
//
//  gamma.cpp -- computation of gamma and log gamma functions.
//      Algorithms and coefficient values from "Computation of Special
//      Functions", Zhang and Jin, John Wiley and Sons, 1996.
//
//  (C) 2003, C. Bond. All rights reserved.

#include <cmath>
#include <stdexcept>

#include <boost/throw_exception.hpp>

namespace npdivs {

using namespace std;


// Returns gamma function of argument 'x'.
//
// NOTE: Throws domain_error if argument is a nonpositive integer or exceeds 171
double gamma(double x)
{
    int i,k,m;
    double ga,gr,r,z;

    static double g[] = {
        1.0,
        0.5772156649015329,
        -0.6558780715202538,
        -0.420026350340952e-1,
        0.1665386113822915,
        -0.421977345555443e-1,
        -0.9621971527877e-2,
        0.7218943246663e-2,
        -0.11651675918591e-2,
        -0.2152416741149e-3,
        0.1280502823882e-3,
        -0.201348547807e-4,
        -0.12504934821e-5,
        0.1133027232e-5,
        -0.2056338417e-6,
        0.6116095e-8,
        0.50020075e-8,
        -0.11812746e-8,
        0.1043427e-9,
        0.77823e-11,
        -0.36968e-11,
        0.51e-12,
        -0.206e-13,
        -0.54e-14,
        0.14e-14};

    if (x > 171.0)
        BOOST_THROW_EXCEPTION(domain_error(
                    "value too large for gamma; use lgamma instead"));
        
    if (x == (int) x) {
        if (x > 0.0) {
            // gamma(n) = (n-1)!
            ga = 1.0;
            for (i=2; i<x; i++) {
                ga *= i;
            }
        }
        else
            BOOST_THROW_EXCEPTION(domain_error(
                        "gamma not defined for nonpositive integers"));

    } else if (x > 0.0 && x - .5 == (int) x) {
        // Gamma(n+1/2) = sqrt(pi) * choose(n-1/2, n) * n!
        //              = sqrt(pi) * (1-1/2) * (2-1/2) * ... * (n-1/2)
        static const double sqrt_pi = std::sqrt(M_PI);

        ga = sqrt_pi;
        for (i = 1; i < x; i++) {
            ga *= i - .5;
        }

    } else {
        if (fabs(x) > 1.0) {
            z = fabs(x);
            m = (int)z;
            r = 1.0;
            for (k=1; k<=m; k++) {
                r *= (z-k);
            }
            z -= m;
        }
        else
            z = x;

        gr = g[24];

        for (k=23; k>=0; k--) {
            gr = gr*z + g[k];
        }
        ga = 1.0/(gr*z);

        if (fabs(x) > 1.0) {
            ga *= r;
            if (x < 0.0) {
                ga = -M_PI/(x*ga*sin(M_PI*x));
            }
        }
    }
    return ga;
}


double lgamma(double x) {
    double x0, x2, xp, gl, gl0;
    int n, k;
    static double a[] = {
        8.333333333333333e-02,
       -2.777777777777778e-03,
        7.936507936507937e-04,
       -5.952380952380952e-04,
        8.417508417508418e-04,
       -1.917526917526918e-03,
        6.410256410256410e-03,
       -2.955065359477124e-02,
        1.796443723688307e-01,
       -1.39243221690590};
    
    x0 = x;
    if (x <= 0.0)
        BOOST_THROW_EXCEPTION(domain_error(
                    "lgamma() not defined for nonpositive arguments"));

    else if ((x == 1.0) || (x == 2.0))
        return 0.0;

    else if (x <= 7.0) {
        n = (int) (7 - x);
        x0 = x + n;
    }

    x2 = 1.0 / (x0*x0);
    xp = 2.0 * M_PI;
    gl0 = a[9];
    for (k=8;k>=0;k--) {
        gl0 = gl0*x2 + a[k];
    }

    gl = gl0/x0 + 0.5*log(xp) + (x0-0.5)*log(x0) - x0;

    if (x <= 7.0) {
        for (k=1; k <= n; k++) {
            gl -= log(x0 - 1.0);
            x0 -= 1.0;
        }
    }
    return gl;
}

}
