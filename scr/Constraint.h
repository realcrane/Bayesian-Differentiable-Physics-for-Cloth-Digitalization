#pragma once

#include "Mesh.h"

using MeshGrad = std::map<Node*, Tensor>;
using MeshHess = std::map<std::pair<Node*, Node* >, Tensor>;

struct Constraint
{
	virtual ~Constraint() {}

	virtual Tensor value(int* sign = nullptr) = 0;
    virtual MeshGrad gradient() = 0;
    virtual MeshGrad project() = 0;

    // energy function
    virtual Tensor energy(Tensor value) = 0;
    virtual Tensor energy_grad(Tensor value) = 0;
    virtual Tensor energy_hess(Tensor value) = 0;

    // frictional force
    virtual MeshGrad friction(Tensor dt, MeshHess& jac) = 0;
};

struct EqCon : public Constraint {
    // n . (node->x - x) = 0
    Node* node;
    Tensor x, n;
    Tensor stiff;

    Tensor value(int* sign = nullptr) override;
    MeshGrad gradient() override;
    MeshGrad project() override;

    Tensor energy(Tensor value) override;
    Tensor energy_grad(Tensor value) override;
    Tensor energy_hess(Tensor value) override;

    MeshGrad friction(Tensor dt, MeshHess& jac) override;
};

struct GlueCon : public Constraint {
    Node* nodes[2];
    Tensor n;
    Tensor stiff;

    Tensor value(int* sign = nullptr) override;
    MeshGrad gradient() override;
    MeshGrad project() override;

    Tensor energy(Tensor value) override;
    Tensor energy_grad(Tensor value) override;
    Tensor energy_hess(Tensor value) override;

    MeshGrad friction(Tensor dt, MeshHess& jac) override;
};

struct IneqCon : public Constraint {
    // n . sum(w[i] verts[i]->x) >= 0
    Node* nodes[4];
    Tensor w[4];
    bool free[4];
    Tensor n;
    Tensor a; // area
    Tensor mu; // friction
    Tensor stiff;

    Tensor value(int* sign = nullptr) override;
    MeshGrad gradient() override;
    MeshGrad project() override;

    Tensor energy(Tensor value) override;
    Tensor energy_grad(Tensor value) override;
    Tensor energy_hess(Tensor value) override;

    MeshGrad friction(Tensor dt, MeshHess& jac) override;
};
