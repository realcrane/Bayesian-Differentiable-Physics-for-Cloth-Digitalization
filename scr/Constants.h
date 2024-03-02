#pragma once
#include <torch/torch.h>

using torch::Tensor;

static const torch::TensorOptions TNOPT = torch::dtype(torch::kFloat64);
static const torch::TensorOptions TNOPT_CUDA = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);

static const Tensor ZERO = torch::zeros({}, TNOPT);
static const Tensor ZERO2 = torch::zeros({ 2 }, TNOPT);
static const Tensor ZERO3 = torch::zeros({ 3 }, TNOPT);
static const Tensor ZERO33 = torch::zeros({ 3, 3 }, TNOPT);

static const Tensor ONE = torch::ones({}, TNOPT);
static const Tensor INFINITY_TENSOR = std::numeric_limits<double>::infinity() * ONE;
static const Tensor EYE3 = torch::eye(3, TNOPT);
static const Tensor EYE3B = torch::eye(3, torch::kBool);

struct Magic {
    bool fixed_high_res_mesh;
    Tensor handle_stiffness, collision_stiffness;
    Tensor repulsion_thickness, projection_thickness;
    Tensor edge_flip_threshold;
    Tensor rib_stiffening;
    bool combine_tensors;
    bool preserve_creases;
    Magic() :
        fixed_high_res_mesh(false),
        handle_stiffness(ONE * 1e3),
        collision_stiffness(ONE * 1e9),
        repulsion_thickness(ONE * 5e-3),
        projection_thickness(ONE * 1e-4),
        edge_flip_threshold(ONE * 1e-2),
        rib_stiffening(ONE),
        combine_tensors(true),
        preserve_creases(false) {}
};


static const Magic magic;