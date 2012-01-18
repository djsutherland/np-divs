#ifndef DIV_ALPHA_HPP_
#define DIV_ALPHA_HPP_
#include "basics.hpp"

#include <vector>

#include "div_func.hpp"

class DivAlpha : public DivFunc {
    typedef DivFunc super;

    protected:
        double alpha;

    public:
        DivAlpha(double alpha=.999, double ub = .99);

        virtual double operator()(
                const std::vector<float> &rho_x,
                const std::vector<float> &nu_x,
                int y_size,
                int dim,
                int k
            ) const;

        virtual double operator()(
                const std::vector<float> &rho_x,
                const std::vector<float> &nu_x,
                const std::vector<float> &rho_y,
                const std::vector<float> &nu_y,
                int dim,
                int k
            ) const;

    private:
        virtual DivAlpha* do_clone() const;
};
#endif
