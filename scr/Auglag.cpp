#include <omp.h>
#include <math.h>

#include "Opti.h"
#include "optimization.h"

using namespace alglib;

static double mu = 1e3;

struct GradInfo
{
    std::vector<double> lambda;
    NLConOpt* problem;
    GradInfo(NLConOpt* p) : problem{ p } {
        lambda = std::vector<double>(p->ncon, 0);
    }

};

static void auglag_value_and_grad(const real_1d_array& x, double& value, real_1d_array& grad, void* ptr = NULL);

static void multiplier_update(const real_1d_array& x, GradInfo& grad_info);

std::vector<double> augmented_lagrangian_method(NLConOpt& problem, OptOptions opt, bool verbose) {


    GradInfo prob_info(&problem);
    real_1d_array x;
    x.setlength(prob_info.problem->nvar);
    prob_info.problem->initialize(&x[0]);


    mincgstate state;
    mincgreport rep;
    mincgcreate(x, state);
    const int max_total_iter = opt.max_iter(),
        max_sub_iter = sqrt(max_total_iter);

    int iter = 0;
    while (iter < max_total_iter) 
    {
        int max_iter = std::min(max_sub_iter, max_total_iter - iter);
        mincgsetcond(state, opt.eps_g(), opt.eps_f(), opt.eps_x(), max_iter);
        if (iter > 0)
            mincgrestartfrom(state, x);
        mincgsuggeststep(state, 1e-3 * prob_info.problem->nvar);
        mincgoptimize(state, auglag_value_and_grad, nullptr, &prob_info);
        mincgresults(state, x, rep);
        multiplier_update(x, prob_info);
        if (verbose)
            std::cout << rep.iterationscount << " iterations" << std::endl;
        if (rep.iterationscount == 0)
            break;
        iter += rep.iterationscount;
    }

    prob_info.problem->finalize(&x[0]);
    return prob_info.lambda;
}

static void add(real_1d_array& x, const std::vector<double>& y) {
    for (int i = 0; i < y.size(); i++)
        x[i] += y[i];
}

inline double clamp_violation(double x, int sign) {
    return (sign < 0) ? std::max(x, 0.) : (sign > 0) ? std::min(x, 0.) : x;
}

static void auglag_value_and_grad(const real_1d_array& x, double& value, real_1d_array& grad, void* ptr) 
{
    GradInfo* temp_info = reinterpret_cast<GradInfo*>(ptr);
    
    temp_info->problem->precompute(&x[0]);
    value = temp_info->problem->objective(&x[0]);
    temp_info->problem->obj_grad(&x[0], &grad[0]);

    double values = 0.0;
    std::vector<double> grads(temp_info->problem->nvar, 0.0);

    for (int j = 0; j < temp_info->problem->ncon; j++) {
        int sign;
        double gj = temp_info->problem->constraint(&x[0], j, sign);
        double cj = clamp_violation(gj + temp_info->lambda[j] / ::mu, sign);
        if (cj != 0) {
            values += ::mu / 2 * (cj * cj);
            temp_info->problem->con_grad(&x[0], j, ::mu * cj, &grads[0]);
        }
    }

    value += values;

    for (int i = 0; i < temp_info->problem->nvar; i++) {
        grad[i] += grads[i];
    }

}

static void multiplier_update(const real_1d_array& x, GradInfo& grad_info)
{
    grad_info.problem->precompute(&x[0]);

    for (int j = 0; j < grad_info.problem->ncon; j++) {
        int sign;
        double gj = grad_info.problem->constraint(&x[0], j, sign);
        grad_info.lambda[j] = clamp_violation(grad_info.lambda[j] + ::mu * gj, sign);
    }
}