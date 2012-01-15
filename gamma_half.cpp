#define _USE_MATH_DEFINES
#include <cmath>
#include <stdexcept>

#include "gamma_half.hpp"

using std::domain_error;

// Factorial of a nonnegative integer, computed the naive way.
inline unsigned long factorial(unsigned int n) {
    unsigned long res = 1;
    for (unsigned int i = 2; i <= n; i++)
        res *= i;
    return res;
}

// Returns gamma(n/2.)
double gamma_half(unsigned int two_n) {
    if (two_n == 0)
        throw domain_error("gamma(0) undefined");

    unsigned int n = two_n / 2;


    if (two_n % 2 == 0) {
        // Gamma(n) = (n-1)!
        return factorial(n - 1);

    } else {
        // Gamma(n+1/2) = sqrt(pi) * choose(n-1/2, n) * n!
        //              = sqrt(pi) * (1-1/2) * (2-1/2) * ... * (n-1/2)
        static const double sqrt_pi = std::sqrt(M_PI);

        double res = sqrt_pi;
        for (unsigned int i = 1; i <= n; i++)
            res *= i - .5;
        return res;
    }
}
