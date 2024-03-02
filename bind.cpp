#include <torch/extension.h>

#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

#include "scr/Simulation.h"
#include "scr/Cloth.h"
#include "scr/MeshIO.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<Cloth>);
PYBIND11_MAKE_OPAQUE(std::vector<Obstacle>);
PYBIND11_MAKE_OPAQUE(std::vector<Handle*>);
PYBIND11_MAKE_OPAQUE(std::vector<Node*>);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

	py::class_<Simulation>(m, "Simulation")
		.def(py::init<>())
		.def_readwrite("dev_wordy", &Simulation::dev_wordy, py::return_value_policy::reference)
		.def_readwrite("device_idx", &Simulation::device_idx, py::return_value_policy::reference)
		.def_readwrite("cloths", &Simulation::cloths, py::return_value_policy::reference)
		.def_readwrite("gravity", &Simulation::gravity, py::return_value_policy::reference)
		.def_readwrite("step_time", &Simulation::step_time, py::return_value_policy::reference)
		.def_readwrite("step_time_cuda", &Simulation::step_time_cuda, py::return_value_policy::reference)
		.def("prepare", &Simulation::prepare, "Prepare Simulation")
		.def("prepare_cuda", &Simulation::prepare_cuda, "Prepare Simulation (GPU)")
		.def("clean_mesh", &Simulation::clean_mesh, "Clean Cloths Obstacles")
		.def("init_vectorization", &Simulation::init_vectorization, "Initialize tensor for vectorization")
		.def("init_vectorization_cuda", &Simulation::init_vectorization_cuda, "Initialize tensor for vectorization (GPU)")
		.def("advance_step", py::overload_cast<>(&Simulation::advance_step), "Simulation Forward")
		.def("advance_step", py::overload_cast<const Tensor&, const Tensor&>(&Simulation::advance_step), "Simulation Forward")
		.def("advance_step_cuda", py::overload_cast<>(&Simulation::advance_step_cuda), "Simulation Forward (GPU)")
		.def("advance_step_cuda", py::overload_cast<const Tensor&, const Tensor&>(&Simulation::advance_step_cuda), "Simulation Forward for checkpoint (GPU)");

	py::class_<Cloth>(m, "Cloth")
		.def(py::init<>())
		.def_readwrite("mesh", &Cloth::mesh, py::return_value_policy::reference)
		.def_readwrite("handle_stiffness_cuda", &Cloth::handle_stiffness_cuda, py::return_value_policy::reference)
		.def("set_parameters", py::overload_cast<const Tensor&, const Tensor&, const Tensor&, const Tensor&>(&Cloth::set_parameters), "Set cloth physical parameter (HOMO)")
		.def("set_parameters", py::overload_cast<const std::vector<Tensor>&, const std::vector<Tensor>&, const std::vector<Tensor>&, const std::vector<Tensor>&>(&Cloth::set_parameters), "Set cloth physical parameter (HETER)")
		.def("set_parameters_cuda", py::overload_cast<const Tensor&, const Tensor&, const Tensor&, const Tensor&>(&Cloth::set_parameters_cuda), "Set cloth physical parameter (HOMO) GPU")
		.def("set_parameters_cuda", py::overload_cast<const std::vector<Tensor>&, const std::vector<Tensor>&, const std::vector<Tensor>&, const std::vector<Tensor>&>(&Cloth::set_parameters_cuda), "Set cloth physical parameter (HETER) GPU")
		.def("set_densities_cuda", &Cloth::set_densities_cuda, "Set cloth density (HETER) GPU")
		.def("set_stretches_cuda", py::overload_cast<const std::vector<Tensor>&>(&Cloth::set_stretches_cuda), "Set cloth stretching parameters (HETER) GPU")
		.def("set_stretches_cuda", py::overload_cast<const std::vector<Tensor>&, const std::vector<Tensor>&, const std::vector<Tensor>&, const std::vector<Tensor>&>(&Cloth::set_stretches_cuda), "Set cloth stretching parameters (each component individually) (HETER) GPU")
		.def("set_bendings_cuda", &Cloth::set_bendings_cuda, "Set cloth bending parameters (HETER) GPU")
		.def("compute_masses", &Cloth::compute_masses, "Compute Cloth Mesh Face and Vertices Mass")
		.def("compute_masses_cuda", &Cloth::compute_masses_cuda, "Compute nodes's mass and general mass matrix (GPU)")
		.def("get_pos_vel", &Cloth::get_pos_vel, py::return_value_policy::reference, "Get cloth position vector and velocity vector")
		.def("update_pos_vel_cuda", &Cloth::update_pos_vel_cuda, py::return_value_policy::reference, "Update cloth position vector and velocity vector (GPU)")
		.def("update_mesh_cuda", &Cloth::update_mesh_cuda, py::return_value_policy::reference, "Update cloth mesh from current position and velocity vector (GPU)")
		.def("get_pos_vel_cuda", &Cloth::get_pos_vel_cuda, py::return_value_policy::reference, "Get cloth position vector and velocity vector (GPU)")
		.def("update_mesh", &Cloth::update_mesh, "Update cloth mesh state for using checkpoint")
		.def("faces_uv_idx", &Cloth::faces_uv_idx, py::return_value_policy::move, "Get cloth mesh face uv and indices");

	py::class_<Mesh>(m, "Mesh")
		.def(py::init<>())
		.def_readwrite("faces", &Mesh::faces, py::return_value_policy::reference)
		.def_readwrite("edges", &Mesh::edges, py::return_value_policy::reference)
		.def_readwrite("verts", &Mesh::verts, py::return_value_policy::reference)
		.def_readwrite("nodes", &Mesh::nodes, py::return_value_policy::reference);

	py::class_<Node>(m, "Node")
		.def_readwrite("x", &Node::x, py::return_value_policy::reference)
		.def_readwrite("v", &Node::v, py::return_value_policy::reference);

	py::class_<Face>(m, "Face")
		.def_readwrite("label", &Face::label, py::return_value_policy::reference);

	m.def("num_bend_edges", &num_bending_edge, "Count bending edges");

	m.def("read_obj", &read_obj, "Read Obj File");
	m.def("save_obj", &save_obj, "Save Obj File");

	py::bind_vector<std::vector<Node*>>(m, "Vector Node");
	py::bind_vector<std::vector<Cloth>>(m, "Vector Cloth");
	py::bind_vector<std::vector<Obstacle>>(m, "Vector Obstacle");
	py::bind_vector<std::vector<Handle*>>(m, "Vector Handle");
}

