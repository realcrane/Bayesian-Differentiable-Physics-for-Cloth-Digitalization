#pragma once

#include <vector>

#include "Mesh.h"

struct Material
{
	Tensor density;
	Tensor stretching;
	Tensor bending;

	Tensor damping;
};

struct Cloth
{
public:

	Mesh mesh;	// Cloth Mesh

	Tensor grid;

	Tensor SF_T, SJ_T;
	Tensor BF_T, BJ_T;

	Tensor face_indices_1, face_indices_2;
	Tensor faces_invDm;

	Tensor faces_m;

	Tensor Du, Dv;

	Tensor bendings_node0, bendings_node1, bendings_node2, bendings_node3;

	Tensor theta_ideals;

	Tensor edges_adf_a, edges_ldaa, edges_bang;

	Tensor edges_indices_t;

	Tensor edges_ajf_indices_1, edges_ajf_indices_2;

	Tensor handle_indices_t, init_pos_t;	// Vectorize handle constrain

	std::vector<Tensor> density;

	Tensor stretch_ori, bending, damping;

	Tensor M;

	Tensor nodes_gravity;

	Tensor handle_stiffness_cuda;
	
	Tensor grid_cuda;

	Tensor SF_T_cuda, SJ_T_cuda;

	Tensor BF_T_cuda, BJ_T_cuda;

	Tensor node_m_matrix;

	Tensor face_indices_1_cuda, face_indices_2_cuda;

	Tensor faces_m_cuda, faces_invDm_cuda;

	Tensor Du_cuda, Dv_cuda;

	Tensor M_cuda;

	Tensor stretch_ori_cuda, bending_cuda, damping_cuda;

	std::vector<Tensor> c11_cuda, c12_cuda, c22_cuda, c33_cuda;

	Tensor gravity_cuda;

	Tensor handle_indices_t_cuda, init_pos_t_cuda;

	Tensor edges_ajf_indices_1_cuda, edges_ajf_indices_2_cuda;

	Tensor bendings_node0_cuda, bendings_node1_cuda, bendings_node2_cuda, bendings_node3_cuda;

	Tensor theta_ideals_cuda;

	Tensor edges_indices_t_cuda;

	Tensor edges_adf_a_cuda, edges_l_cuda, edges_ldaa_cuda, edges_bang_cuda;

	Tensor pos_cuda, vel_cuda;

	void compute_masses();

	void compute_masses_cuda();

	void set_parameters(const Tensor& density, const Tensor& stretching, const Tensor& bending, const Tensor& damping);

	void set_parameters(const std::vector<Tensor>& density, const std::vector<Tensor>& stretching, const std::vector<Tensor>& bending, const std::vector<Tensor>& damping);

	void set_parameters_cuda(const Tensor& density, const Tensor& stretching, const Tensor& bending, const Tensor& damping);

	void set_parameters_cuda(const std::vector<Tensor>& density, const std::vector<Tensor>& stretching, const std::vector<Tensor>& bending, const std::vector<Tensor>& damping);

	void set_densities_cuda(const std::vector<Tensor>& density);

	void set_stretches_cuda(const std::vector<Tensor>& stretching);

	void set_stretches_cuda(const std::vector<Tensor>& c11s, const std::vector<Tensor>& c12s, const std::vector<Tensor>& c22s, const std::vector<Tensor>& c33s);
	
	void set_bendings_cuda(const std::vector<Tensor>& bending);

	std::pair<Tensor, Tensor> get_pos_vel();

	std::pair<Tensor, Tensor> get_pos_vel_cuda();

	void update_pos_vel_cuda(const int& device_idx);

	void update_mesh_cuda(const Tensor& pos, const Tensor& vel);

	std::pair<Tensor, Tensor> faces_uv_idx();

	void update_mesh(const Tensor& pos, const Tensor& vel);
};