#pragma once
#include "torch/extension.h"
#include "Cloth.h"
#include "Simulation.h"
#include "Solve.h"

using torch::Tensor;

void add_noise_forces(const Cloth& cloth, const Tensor& noise_face, Tensor& fext, Tensor& Jext);

void add_external_forces(const Cloth& cloth, const Tensor& gravity, const Wind& wind, std::vector<Tensor>& fext, std::vector<Tensor>& Jext);

void add_constraint_forces(const Cloth& cloth, const std::vector<Constraint*>& cons, SparseMatrix& A, std::vector<Tensor>& b, Tensor dt);

void implicit_update(Cloth& cloth, std::vector<Handle*> handles, const Tensor& dt);

void implicit_update(Cloth& cloth, const Tensor& ns_t, const Tensor& vs_t, const Tensor& dt);

void implicit_update_cuda(Cloth& cloth, const Tensor& dt_cpu, const Tensor& dt_cuda);

void implicit_update_cuda(Cloth& cloth, const Tensor& ns_t_cuda, const Tensor& vs_t_cuda, const Tensor& dt, const int& device_idx, const torch::TensorOptions& tensor_opt);
