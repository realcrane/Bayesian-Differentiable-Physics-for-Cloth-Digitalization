#include "Constraint.h"


Tensor EqCon::value(int* sign) 
{
    if (sign != nullptr)
    {
        *sign = 0;
    }

    return torch::dot(n, node->x - x);

}

MeshGrad EqCon::gradient() { MeshGrad grad; grad[node] = n; return grad; }
MeshGrad EqCon::project() { return MeshGrad(); }
Tensor EqCon::energy(Tensor value) { return stiff * (value * value) / 2.; }
Tensor EqCon::energy_grad(Tensor value) { return stiff * value; }
Tensor EqCon::energy_hess(Tensor value) { return stiff; }
MeshGrad EqCon::friction(Tensor dt, MeshHess& jac) { return MeshGrad(); }

Tensor GlueCon::value(int* sign) {

    if (sign != nullptr)
    {
        *sign = 0;
    }

    return torch::dot(n, nodes[1]->x - nodes[0]->x);
}

MeshGrad GlueCon::gradient() 
{
    MeshGrad grad;
    grad[nodes[0]] = -n;
    grad[nodes[1]] = n;

    return grad;
}

MeshGrad GlueCon::project() { return MeshGrad(); }
Tensor GlueCon::energy(Tensor value) { return stiff * (value * value) / 2.; }
Tensor GlueCon::energy_grad(Tensor value) { return stiff * value; }
Tensor GlueCon::energy_hess(Tensor value) { return stiff; }
MeshGrad GlueCon::friction(Tensor dt, MeshHess& jac) { return MeshGrad(); }

Tensor IneqCon::value(int* sign) {
    if (sign)
        *sign = 1;
    Tensor d = ZERO;
    for (int i = 0; i < 4; i++)
        d = d + w[i] * dot(n, nodes[i]->x);
    d = d - magic.repulsion_thickness;
    return detach(d);
}

MeshGrad IneqCon::gradient() {
    MeshGrad grad;
    for (int i = 0; i < 4; i++)
        grad[nodes[i]] = detach(w[i] * n);
    return grad;
}

MeshGrad IneqCon::project() {
    Tensor d = value() + magic.repulsion_thickness - magic.projection_thickness;
    if ((d >= 0).item<int>())
        return MeshGrad();
    Tensor inv_mass = ZERO;
    for (int i = 0; i < 4; i++)
        if (free[i])
            inv_mass = inv_mass + (w[i] * w[i]) / nodes[i]->m;
    MeshGrad dx;
    for (int i = 0; i < 4; i++)
        if (free[i])
            dx[nodes[i]] = detach(-(w[i] / nodes[i]->m) / inv_mass * n * d);
    return dx;
}

Tensor violation(Tensor value) 
{ 
    return torch::max(-value, ZERO); 
}

Tensor IneqCon::energy(Tensor value) {
    Tensor v = violation(value);
    return stiff * v * v * v / magic.repulsion_thickness / 6;
}

Tensor IneqCon::energy_grad(Tensor value) {
    return -stiff * (violation(value) * violation(value)) / magic.repulsion_thickness / 2;
}

Tensor IneqCon::energy_hess(Tensor value) {
    return stiff * violation(value) / magic.repulsion_thickness;
}

MeshGrad IneqCon::friction(Tensor dt, MeshHess& jac) {
    if ((mu == 0).item<int>())
        return MeshGrad();
    Tensor fn = abs(energy_grad(value()));
    if ((fn == 0).item<int>())
        return MeshGrad();
    Tensor v = ZERO3;
    Tensor inv_mass = ZERO;
    for (int i = 0; i < 4; i++) {
        v = v + w[i] * nodes[i]->v;
        if (free[i])
            inv_mass = inv_mass + (w[i] * w[i]) / nodes[i]->m;
    }
    Tensor T = torch::eye(3, TNOPT) - ger(n, n);
    Tensor vt = norm(matmul(T, v));
    Tensor f_by_v = min(mu * fn / vt, 1 / (dt * inv_mass));
    // double f_by_v = mu*fn/max(vt, 1e-1);
    MeshGrad force;
    for (int i = 0; i < 4; i++) {
        if (free[i]) {
            force[nodes[i]] = detach(-w[i] * f_by_v * matmul(T, v));
            for (int j = 0; j < 4; j++) {
                if (free[j]) {
                    jac[std::make_pair(nodes[i], nodes[j])] = detach(-w[i] * w[j] * f_by_v * T);
                }
            }
        }
    }
    return force;
}