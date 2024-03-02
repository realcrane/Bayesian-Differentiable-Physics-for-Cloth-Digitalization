#pragma once
#include <torch/extension.h>

#include "Mesh.h"

using torch::Tensor;

Tensor stretching_stiffness(const Tensor& G, const Tensor samples);

Tensor batch_stretching_stiffness(const Tensor& G, const Tensor& samples);

Tensor batch_stretching_stiffness_cuda(const Tensor& G, const Tensor& samples);

Tensor bending_stiffness(const Edge* edge, const Tensor& data0, const Tensor& data1);

Tensor batch_bending_stiffness(Tensor curv, Tensor bang, Tensor bend);

Tensor batch_bending_stiffness_cuda(const Tensor& curv, const Tensor& bang, const Tensor& bend);

Tensor batch_bending_stiffness(Tensor curv, Tensor bang, Tensor bend0, Tensor bend1);

Tensor evaluate_stretching_samples(const Tensor& stretch_ori, const Tensor& grid);
