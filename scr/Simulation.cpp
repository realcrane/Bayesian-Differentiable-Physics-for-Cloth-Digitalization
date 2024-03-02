#include "Simulation.h"
#include "Physics.h"
#include "Proximity.h"

#include <chrono>

using namespace std::chrono;

void Simulation::prepare()
{
	if (dev_wordy)
		std::cout << "Prepare Cloth and Obstacle Meshes" << std::endl;
	
	step = 0;
	frame = 0;

	// Initialize Cloth Mesh
	for (auto& c : cloths)
	{
		c.compute_masses();
		c.mesh.update_x0();
	}

	// Initialize Obstacle Mesh
	for (auto& o : obstacles)
	{		
		o.mesh.update_x0();
	}
}

void Simulation::prepare_cuda()
{
	if (dev_wordy)
		std::cout << "Prepare Cloth and Obstacle Meshes" << std::endl;
	
	step = 0;
	frame = 0;

	// Initialize Cloth Mesh
	for (auto& c : cloths)
	{
		c.compute_masses_cuda();
		c.mesh.update_x0();
	}

	// Initialize Obstacle Mesh
	for (auto& o : obstacles)
	{		
		o.mesh.update_x0();
	}
}

void Simulation::init_vectorization()
{
	int num_nodes{ static_cast<int>(cloths[0].mesh.nodes.size()) };
	int num_edges{ static_cast<int>(cloths[0].mesh.edges.size()) };
	int num_faces{ static_cast<int>(cloths[0].mesh.faces.size()) };

	std::vector<Tensor> nodes_gravity;

	std::vector<double> input_grid;

	std::vector<Tensor> tris_indices_1, tris_indices_2;
	std::vector<Tensor> faces_m, faces_invDm;
	std::vector<Tensor> Dus, Dvs;
	std::vector<Tensor> stretch_f_tensors;
	std::vector<Tensor> stretch_J_tensors;

	std::vector<Tensor> bendings_nodes_0, bendings_nodes_1, bendings_nodes_2, bendings_nodes_3;
	std::vector<Tensor> theta_ideals;
	std::vector<Tensor> edges_indices;
	std::vector<Tensor> edges_adf_a, edges_ldaa, edges_bang;
	std::vector<Tensor> edges_ajf_indices_1, edges_ajf_indices_2;
	std::vector<Tensor> bending_f_tensors;
	std::vector<Tensor> bending_J_tensors;

	for (int n = 0; n < num_nodes; ++n) {
		nodes_gravity.push_back(gravity);
	}

	cloths[0].nodes_gravity = torch::cat(nodes_gravity);

	int nsamples{ 30 };

	for (int i = 0; i < nsamples; ++i) {
		for (int j = 0; j < nsamples; ++j) {
			for (int k = 0; k < nsamples; ++k) {
				double a, b, c;
				a = 0.5 + i * 2 / (nsamples * 1.0);
				b = 0.5 + j * 2 / (nsamples * 1.0);
				c = k * 2 / (nsamples * 1.0);

				double w = 0.5 * (a + b + std::sqrt(4 * std::pow(c, 2) + std::pow(a - b, 2)));

				double v1 = c;
				double v0 = w - b;

				if (k == 0) {
					if (i >= j) {
						v1 = 0;
						v0 = 1;
					}
					else {
						v1 = 1;
						v0 = 0;
					}
				}

				double angle_weight = std::fabs(std::atan2(v1, v0) / M_PI) * 8;
				double strain_weight = (std::sqrt(w) - 1) * 6;

				input_grid.push_back(angle_weight/2 - 1);
				input_grid.push_back(strain_weight*2 - 1);
			}
		}
	}

	cloths[0].grid = torch::tensor(input_grid, TNOPT).reshape({ 1, nsamples * nsamples, nsamples, 2 });

	for (int f = 0; f < num_faces; ++f)
	{
		if (f % 100 == 0) {
			std::cout << "Processing Face: " << f << "/" << num_faces << std::endl;
		}

		const Face* face = cloths[0].mesh.faces[f];

		int n0_idx = face->v[0]->node->index;
		int n1_idx = face->v[1]->node->index;
		int n2_idx = face->v[2]->node->index;

		Tensor tri_values = torch::tensor({ 1, 1, 1, -1, -1, -1 }, torch::kInt8);

		Tensor tri_indices_row = torch::arange(3, torch::kLong).repeat(2);
		Tensor tri_1_indices_col = torch::tensor({ n1_idx * 3 , n0_idx * 3 }, torch::kLong).repeat_interleave(3) + torch::arange(3, torch::kLong).repeat(2);
		Tensor tri_2_indices_col = torch::tensor({ n2_idx * 3 , n0_idx * 3 }, torch::kLong).repeat_interleave(3) + torch::arange(3, torch::kLong).repeat(2);

		Tensor tri_1_spa_indices = torch::stack({ tri_indices_row, tri_1_indices_col }, 0);
		Tensor tri_2_spa_indices = torch::stack({ tri_indices_row, tri_2_indices_col }, 0);

		auto tri_1_indices = torch::sparse_coo_tensor(tri_1_spa_indices, tri_values, { 3, num_nodes * 3 }, torch::kInt8);
		auto tri_2_indices = torch::sparse_coo_tensor(tri_2_spa_indices, tri_values, { 3, num_nodes * 3 }, torch::kInt8);

		tris_indices_1.push_back(tri_1_indices);
		tris_indices_2.push_back(tri_2_indices);

		faces_invDm.push_back(cloths[0].mesh.faces[f]->invDm);

		faces_m.push_back(face->m_a);

		Tensor d = face->invDm.flatten();
		Tensor Du = torch::cat({ (-d[0] - d[2]) * EYE3, d[0] * EYE3, d[2] * EYE3 }, 1);
		Tensor Dv = torch::cat({ (-d[1] - d[3]) * EYE3, d[1] * EYE3, d[3] * EYE3 }, 1);

		Dus.push_back(Du);
		Dvs.push_back(Dv);

		Tensor stretch_f_tensor = torch::zeros({ num_nodes * 3, 9 }, torch::kInt8);

		stretch_f_tensor.index_put_({ Slice(n0_idx * 3, n0_idx * 3 + 3), Slice(0, 3) }, torch::eye(3, torch::kInt8));
		stretch_f_tensor.index_put_({ Slice(n1_idx * 3, n1_idx * 3 + 3), Slice(3, 6) }, torch::eye(3, torch::kInt8));
		stretch_f_tensor.index_put_({ Slice(n2_idx * 3, n2_idx * 3 + 3), Slice(6, 9) }, torch::eye(3, torch::kInt8));

		int d_n0_x_d_n0 = n0_idx * (num_nodes * 3 * 3) + n0_idx * 3;
		int d_n0_x_d_n1 = n0_idx * (num_nodes * 3 * 3) + n1_idx * 3;
		int d_n0_x_d_n2 = n0_idx * (num_nodes * 3 * 3) + n2_idx * 3;

		int d_n0_y_d_n0 = n0_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n0_idx * 3;
		int d_n0_y_d_n1 = n0_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n1_idx * 3;
		int d_n0_y_d_n2 = n0_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n2_idx * 3;

		int d_n0_z_d_n0 = n0_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n0_idx * 3;
		int d_n0_z_d_n1 = n0_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n1_idx * 3;
		int d_n0_z_d_n2 = n0_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n2_idx * 3;

		int d_n1_x_d_n0 = n1_idx * (num_nodes * 3 * 3) + n0_idx * 3;
		int d_n1_x_d_n1 = n1_idx * (num_nodes * 3 * 3) + n1_idx * 3;
		int d_n1_x_d_n2 = n1_idx * (num_nodes * 3 * 3) + n2_idx * 3;

		int d_n1_y_d_n0 = n1_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n0_idx * 3;
		int d_n1_y_d_n1 = n1_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n1_idx * 3;
		int d_n1_y_d_n2 = n1_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n2_idx * 3;

		int d_n1_z_d_n0 = n1_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n0_idx * 3;
		int d_n1_z_d_n1 = n1_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n1_idx * 3;
		int d_n1_z_d_n2 = n1_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n2_idx * 3;

		int d_n2_x_d_n0 = n2_idx * (num_nodes * 3 * 3) + n0_idx * 3;
		int d_n2_x_d_n1 = n2_idx * (num_nodes * 3 * 3) + n1_idx * 3;
		int d_n2_x_d_n2 = n2_idx * (num_nodes * 3 * 3) + n2_idx * 3;

		int d_n2_y_d_n0 = n2_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n0_idx * 3;
		int d_n2_y_d_n1 = n2_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n1_idx * 3;
		int d_n2_y_d_n2 = n2_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n2_idx * 3;

		int d_n2_z_d_n0 = n2_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n0_idx * 3;
		int d_n2_z_d_n1 = n2_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n1_idx * 3;
		int d_n2_z_d_n2 = n2_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n2_idx * 3;

		Tensor row_index = torch::tensor({
			d_n0_x_d_n0, d_n0_x_d_n1, d_n0_x_d_n2, d_n0_y_d_n0, d_n0_y_d_n1, d_n0_y_d_n2, d_n0_z_d_n0, d_n0_z_d_n1, d_n0_z_d_n2,
			d_n1_x_d_n0, d_n1_x_d_n1, d_n1_x_d_n2, d_n1_y_d_n0, d_n1_y_d_n1, d_n1_y_d_n2, d_n1_z_d_n0, d_n1_z_d_n1, d_n1_z_d_n2,
			d_n2_x_d_n0, d_n2_x_d_n1, d_n2_x_d_n2, d_n2_y_d_n0, d_n2_y_d_n1, d_n2_y_d_n2, d_n2_z_d_n0, d_n2_z_d_n1, d_n2_z_d_n2
			}, torch::kLong);

		row_index = row_index.repeat_interleave(3) + torch::tensor({ 0, 1, 2 }, torch::kInt).repeat(27);

		Tensor col_index = torch::arange(81, torch::kLong);

		Tensor indices = torch::stack({ row_index, col_index }, 0);

		Tensor values = torch::ones(81, torch::kInt8);

		auto stretch_J_tensor = torch::sparse_coo_tensor(indices, values, { num_nodes * 3 * num_nodes * 3, 3 * 3 * 3 * 3 }, torch::kInt8);

		stretch_f_tensors.push_back(stretch_f_tensor.to_sparse());
		stretch_J_tensors.push_back(stretch_J_tensor);

	}

	cloths[0].face_indices_1 = torch::cat(tris_indices_1, 0).to(TNOPT);
	cloths[0].face_indices_2 = torch::cat(tris_indices_2, 0).to(TNOPT);

	cloths[0].faces_invDm = torch::stack(faces_invDm, 0);

	cloths[0].faces_m = torch::stack(faces_m);

	cloths[0].Du = torch::stack(Dus);
	cloths[0].Dv = torch::stack(Dvs);

	cloths[0].SF_T = torch::cat(stretch_f_tensors, 1).to(TNOPT);
	cloths[0].SJ_T = torch::cat(stretch_J_tensors, 1).to(TNOPT);

	for (int e = 0; e < num_edges; ++e)
	{
		if (e % 100 == 0) {
			std::cout << "Processing Edge: " << e << "/" << num_edges << std::endl;
		}

		const Edge* edge = cloths[0].mesh.edges[e];

		// Rule out the egdes that are not between to faces
		if (edge->adj_faces[0] == nullptr || edge->adj_faces[1] == nullptr)
			continue;

		int n0_idx = edge->nodes[0]->index;
		int n1_idx = edge->nodes[1]->index;
		int n2_idx = edge_opp_vert(edge, 0)->node->index;
		int n3_idx = edge_opp_vert(edge, 1)->node->index;

		edges_ldaa.push_back(edge->ldaa);
		edges_bang.push_back(edge->bias_angle);

		theta_ideals.push_back(edge->theta_ideal);

		Tensor edge_values = torch::tensor({ 1, 1, 1, -1, -1, -1 }, torch::kInt8);

		Tensor edge_indices_row = torch::arange(3, torch::kLong).repeat(2);
		Tensor edge_indices_col = torch::tensor({ n0_idx * 3 , n1_idx * 3 }, torch::kLong).repeat_interleave(3) + torch::arange(3, torch::kLong).repeat(2);

		Tensor edge_spa_indices = torch::stack({ edge_indices_row, edge_indices_col }, 0);

		auto edge_indices = torch::sparse_coo_tensor(edge_spa_indices, edge_values, { 3, num_nodes * 3 }, torch::kInt8);

		edges_indices.push_back(edge_indices);

		int f1_idx = edge->adj_faces[0]->index;
		int f2_idx = edge->adj_faces[1]->index;

		edges_adf_a.push_back(edge->adj_faces[0]->m_a + edge->adj_faces[1]->m_a);

		Tensor node_pos = torch::tensor({ 1, 1, 1 }, torch::kInt8);

		Tensor node_row_idx = torch::arange(3, torch::kLong);

		Tensor edge_ajf_indices_col_1 = torch::tensor({ f1_idx * 3, f1_idx * 3, f1_idx * 3 }, torch::kLong) + torch::arange(3, torch::kLong);
		Tensor edge_ajf_indices_col_2 = torch::tensor({ f2_idx * 3, f2_idx * 3, f2_idx * 3 }, torch::kLong) + torch::arange(3, torch::kLong);

		Tensor edge_ajf_spa_indices_1 = torch::stack({ node_row_idx, edge_ajf_indices_col_1 }, 0);
		Tensor edge_ajf_spa_indices_2 = torch::stack({ node_row_idx, edge_ajf_indices_col_2 }, 0);

		auto edge_ajf_indices_1 = torch::sparse_coo_tensor(edge_ajf_spa_indices_1, node_pos, { 3, num_faces * 3 }, torch::kInt8);
		auto edge_ajf_indices_2 = torch::sparse_coo_tensor(edge_ajf_spa_indices_2, node_pos, { 3, num_faces * 3 }, torch::kInt8);

		edges_ajf_indices_1.push_back(edge_ajf_indices_1);
		edges_ajf_indices_2.push_back(edge_ajf_indices_2);

		Tensor n0_indices_col = torch::tensor({ n0_idx * 3, n0_idx * 3 + 1, n0_idx * 3 + 2 }, torch::kLong);
		Tensor n1_indices_col = torch::tensor({ n1_idx * 3, n1_idx * 3 + 1, n1_idx * 3 + 2 }, torch::kLong);
		Tensor n2_indices_col = torch::tensor({ n2_idx * 3, n2_idx * 3 + 1, n2_idx * 3 + 2 }, torch::kLong);
		Tensor n3_indices_col = torch::tensor({ n3_idx * 3, n3_idx * 3 + 1, n3_idx * 3 + 2 }, torch::kLong);

		Tensor n0_spa_idx = torch::stack({ node_row_idx, n0_indices_col }, 0);
		Tensor n1_spa_idx = torch::stack({ node_row_idx, n1_indices_col }, 0);
		Tensor n2_spa_idx = torch::stack({ node_row_idx, n2_indices_col }, 0);
		Tensor n3_spa_idx = torch::stack({ node_row_idx, n3_indices_col }, 0);

		auto n0_indices_t = torch::sparse_coo_tensor(n0_spa_idx, node_pos, { 3, num_nodes * 3 }, torch::kInt8);
		auto n1_indices_t = torch::sparse_coo_tensor(n1_spa_idx, node_pos, { 3, num_nodes * 3 }, torch::kInt8);
		auto n2_indices_t = torch::sparse_coo_tensor(n2_spa_idx, node_pos, { 3, num_nodes * 3 }, torch::kInt8);
		auto n3_indices_t = torch::sparse_coo_tensor(n3_spa_idx, node_pos, { 3, num_nodes * 3 }, torch::kInt8);

		bendings_nodes_0.push_back(n0_indices_t);
		bendings_nodes_1.push_back(n1_indices_t);
		bendings_nodes_2.push_back(n2_indices_t);
		bendings_nodes_3.push_back(n3_indices_t);

		Tensor bending_f_tensor = torch::zeros({ num_nodes * 3, 12 }, torch::kInt8);

		bending_f_tensor.index_put_({ Slice(n0_idx * 3, n0_idx * 3 + 3), Slice(0, 3) }, torch::eye(3, torch::kInt8));
		bending_f_tensor.index_put_({ Slice(n1_idx * 3, n1_idx * 3 + 3), Slice(3, 6) }, torch::eye(3, torch::kInt8));
		bending_f_tensor.index_put_({ Slice(n2_idx * 3, n2_idx * 3 + 3), Slice(6, 9) }, torch::eye(3, torch::kInt8));
		bending_f_tensor.index_put_({ Slice(n3_idx * 3, n3_idx * 3 + 3), Slice(9, 12) }, torch::eye(3, torch::kInt8));

		int d_n0_x_d_n0 = n0_idx * (num_nodes * 3 * 3) + n0_idx * 3;
		int d_n0_x_d_n1 = n0_idx * (num_nodes * 3 * 3) + n1_idx * 3;
		int d_n0_x_d_n2 = n0_idx * (num_nodes * 3 * 3) + n2_idx * 3;
		int d_n0_x_d_n3 = n0_idx * (num_nodes * 3 * 3) + n3_idx * 3;

		int d_n0_y_d_n0 = n0_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n0_idx * 3;
		int d_n0_y_d_n1 = n0_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n1_idx * 3;
		int d_n0_y_d_n2 = n0_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n2_idx * 3;
		int d_n0_y_d_n3 = n0_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n3_idx * 3;

		int d_n0_z_d_n0 = n0_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n0_idx * 3;
		int d_n0_z_d_n1 = n0_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n1_idx * 3;
		int d_n0_z_d_n2 = n0_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n2_idx * 3;
		int d_n0_z_d_n3 = n0_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n3_idx * 3;

		int d_n1_x_d_n0 = n1_idx * (num_nodes * 3 * 3) + n0_idx * 3;
		int d_n1_x_d_n1 = n1_idx * (num_nodes * 3 * 3) + n1_idx * 3;
		int d_n1_x_d_n2 = n1_idx * (num_nodes * 3 * 3) + n2_idx * 3;
		int d_n1_x_d_n3 = n1_idx * (num_nodes * 3 * 3) + n3_idx * 3;

		int d_n1_y_d_n0 = n1_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n0_idx * 3;
		int d_n1_y_d_n1 = n1_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n1_idx * 3;
		int d_n1_y_d_n2 = n1_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n2_idx * 3;
		int d_n1_y_d_n3 = n1_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n3_idx * 3;

		int d_n1_z_d_n0 = n1_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n0_idx * 3;
		int d_n1_z_d_n1 = n1_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n1_idx * 3;
		int d_n1_z_d_n2 = n1_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n2_idx * 3;
		int d_n1_z_d_n3 = n1_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n3_idx * 3;

		int d_n2_x_d_n0 = n2_idx * (num_nodes * 3 * 3) + n0_idx * 3;
		int d_n2_x_d_n1 = n2_idx * (num_nodes * 3 * 3) + n1_idx * 3;
		int d_n2_x_d_n2 = n2_idx * (num_nodes * 3 * 3) + n2_idx * 3;
		int d_n2_x_d_n3 = n2_idx * (num_nodes * 3 * 3) + n3_idx * 3;

		int d_n2_y_d_n0 = n2_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n0_idx * 3;
		int d_n2_y_d_n1 = n2_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n1_idx * 3;
		int d_n2_y_d_n2 = n2_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n2_idx * 3;
		int d_n2_y_d_n3 = n2_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n3_idx * 3;

		int d_n2_z_d_n0 = n2_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n0_idx * 3;
		int d_n2_z_d_n1 = n2_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n1_idx * 3;
		int d_n2_z_d_n2 = n2_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n2_idx * 3;
		int d_n2_z_d_n3 = n2_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n3_idx * 3;

		int d_n3_x_d_n0 = n3_idx * (num_nodes * 3 * 3) + n0_idx * 3;
		int d_n3_x_d_n1 = n3_idx * (num_nodes * 3 * 3) + n1_idx * 3;
		int d_n3_x_d_n2 = n3_idx * (num_nodes * 3 * 3) + n2_idx * 3;
		int d_n3_x_d_n3 = n3_idx * (num_nodes * 3 * 3) + n3_idx * 3;

		int d_n3_y_d_n0 = n3_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n0_idx * 3;
		int d_n3_y_d_n1 = n3_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n1_idx * 3;
		int d_n3_y_d_n2 = n3_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n2_idx * 3;
		int d_n3_y_d_n3 = n3_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n3_idx * 3;

		int d_n3_z_d_n0 = n3_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n0_idx * 3;
		int d_n3_z_d_n1 = n3_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n1_idx * 3;
		int d_n3_z_d_n2 = n3_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n2_idx * 3;
		int d_n3_z_d_n3 = n3_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n3_idx * 3;

		Tensor row_index = torch::tensor({
			d_n0_x_d_n0, d_n0_x_d_n1, d_n0_x_d_n2, d_n0_x_d_n3,
			d_n0_y_d_n0, d_n0_y_d_n1, d_n0_y_d_n2, d_n0_y_d_n3,
			d_n0_z_d_n0, d_n0_z_d_n1, d_n0_z_d_n2, d_n0_z_d_n3,
			d_n1_x_d_n0, d_n1_x_d_n1, d_n1_x_d_n2, d_n1_x_d_n3,
			d_n1_y_d_n0, d_n1_y_d_n1, d_n1_y_d_n2, d_n1_y_d_n3,
			d_n1_z_d_n0, d_n1_z_d_n1, d_n1_z_d_n2, d_n1_z_d_n3,
			d_n2_x_d_n0, d_n2_x_d_n1, d_n2_x_d_n2, d_n2_x_d_n3,
			d_n2_y_d_n0, d_n2_y_d_n1, d_n2_y_d_n2, d_n2_y_d_n3,
			d_n2_z_d_n0, d_n2_z_d_n1, d_n2_z_d_n2, d_n2_z_d_n3,
			d_n3_x_d_n0, d_n3_x_d_n1, d_n3_x_d_n2, d_n3_x_d_n3,
			d_n3_y_d_n0, d_n3_y_d_n1, d_n3_y_d_n2, d_n3_y_d_n3,
			d_n3_z_d_n0, d_n3_z_d_n1, d_n3_z_d_n2, d_n3_z_d_n3 }, torch::kLong);

		row_index = row_index.repeat_interleave(3) + torch::tensor({ 0, 1, 2 }, torch::kInt).repeat(48);

		Tensor col_index = torch::arange(144, torch::kLong);

		Tensor indices = torch::stack({ row_index, col_index }, 0);

		Tensor values = torch::ones(144, torch::kInt8);

		auto bending_J_tensor = torch::sparse_coo_tensor(indices, values, { num_nodes * 3 * num_nodes * 3, 4 * 3 * 4 * 3 }, torch::kInt8);

		bending_f_tensors.push_back(bending_f_tensor.to_sparse());
		bending_J_tensors.push_back(bending_J_tensor);
	}

	cloths[0].edges_indices_t = torch::cat(edges_indices, 0).to(TNOPT);

	cloths[0].edges_adf_a = torch::stack(edges_adf_a);

	cloths[0].edges_ldaa = torch::stack(edges_ldaa);

	cloths[0].edges_bang = torch::stack(edges_bang);

	cloths[0].theta_ideals = torch::stack(theta_ideals);

	cloths[0].edges_ajf_indices_1 = torch::cat(edges_ajf_indices_1, 0).to(TNOPT);
	cloths[0].edges_ajf_indices_2 = torch::cat(edges_ajf_indices_2, 0).to(TNOPT);

	cloths[0].bendings_node0 = torch::cat(bendings_nodes_0, 0).to(TNOPT);
	cloths[0].bendings_node1 = torch::cat(bendings_nodes_1, 0).to(TNOPT);
	cloths[0].bendings_node2 = torch::cat(bendings_nodes_2, 0).to(TNOPT);
	cloths[0].bendings_node3 = torch::cat(bendings_nodes_3, 0).to(TNOPT);

	cloths[0].BF_T = torch::cat(bending_f_tensors, 1).to(TNOPT);
	cloths[0].BJ_T = torch::cat(bending_J_tensors, 1).to(TNOPT);

	std::vector<int> handle_indices = std::vector<int>(num_nodes, 0);
	std::vector<Tensor> init_pos_vec;

	for (int n = 0; n < num_nodes; ++n)
	{
		auto dist = torch::norm(cloths[0].mesh.nodes[n]->x);

		init_pos_vec.push_back(cloths[0].mesh.nodes[n]->x);

		if (dist.item<double>() < 0.09) {
			handle_indices[n] = 1;
		}
	}

	cloths[0].init_pos_t = torch::cat(init_pos_vec);
	cloths[0].handle_indices_t = torch::diag(torch::tensor(handle_indices, TNOPT).repeat_interleave(3));
}

void Simulation::init_vectorization_cuda()
{
	tensor_opt = torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA, device_idx);

	int num_nodes{ static_cast<int>(cloths[0].mesh.nodes.size()) };
	int num_edges{ static_cast<int>(cloths[0].mesh.edges.size()) };
	int num_faces{ static_cast<int>(cloths[0].mesh.faces.size()) };

	std::vector<Tensor> nodes_gravity;

	std::vector<double> input_grid;

	std::vector<double> node_m(num_nodes * num_faces, 0.0);
	std::vector<Tensor> tris_indices_1, tris_indices_2;
	std::vector<Tensor> faces_m, faces_invDm;
	std::vector<Tensor> Dus, Dvs;
	std::vector<Tensor> stretch_f_tensors;
	std::vector<Tensor> stretch_J_tensors;

	std::vector<Tensor> bendings_nodes_0, bendings_nodes_1, bendings_nodes_2, bendings_nodes_3;
	std::vector<Tensor> theta_ideals;
	std::vector<Tensor> edges_indices;
	std::vector<Tensor> edges_adf_a, edges_l, edges_ldaa, edges_bang;
	std::vector<Tensor> edges_ajf_indices_1, edges_ajf_indices_2;
	std::vector<Tensor> bending_f_tensors;
	std::vector<Tensor> bending_J_tensors;

	for (int n = 0; n < num_nodes; ++n) {
		nodes_gravity.push_back(gravity);
	}

	cloths[0].nodes_gravity = torch::cat(nodes_gravity);

	int nsamples{ 10 };

	for (int i = 0; i < nsamples; ++i) {
		for (int j = 0; j < nsamples; ++j) {
			for (int k = 0; k < nsamples; ++k) {

				double a = 0.5 + i * 2 / (nsamples * 1.0); // There may be a bug -0.5 instead of 0.5
				double b = 0.5 + j * 2 / (nsamples * 1.0); // There may be a bug -0.5 instead of 0.5
				double c = k * 2 / (nsamples * 1.0);

				double w = 0.5 * (a + b + std::sqrt(4 * std::pow(c, 2) + std::pow(a - b, 2)));

				double v1 = c;
				double v0 = w - b;

				if (k == 0) {
					if (i >= j) {
						v1 = 0;
						v0 = 1;
					}
					else {
						v1 = 1;
						v0 = 0;
					}
				}

				double angle_weight = std::fabs(std::atan2(v1, v0) / M_PI) * 8;
				double strain_weight = (std::sqrt(w) - 1) * 6;

				input_grid.push_back(angle_weight / 2.0 - 1.0);
				input_grid.push_back(strain_weight * 2.0 - 1.0);
			}
		}
	}

	cloths[0].grid_cuda = torch::tensor(input_grid, tensor_opt).reshape({ 1, nsamples * nsamples, nsamples, 2 });

	for (int f = 0; f < num_faces; ++f)
	{
		if (f % 100 == 0) {
			std::cout << "Processing Face: " << f << "/" << num_faces << std::endl;
		}

		const Face* face = cloths[0].mesh.faces[f];

		int n0_idx = face->v[0]->node->index;
		int n1_idx = face->v[1]->node->index;
		int n2_idx = face->v[2]->node->index;

		node_m[n0_idx * num_faces + f] += 1.0 / 3.0;
		node_m[n1_idx * num_faces + f] += 1.0 / 3.0;
		node_m[n2_idx * num_faces + f] += 1.0 / 3.0;

		Tensor tri_values = torch::tensor({ 1, 1, 1, -1, -1, -1 }, torch::kInt8);

		Tensor tri_indices_row = torch::arange(3, torch::kLong).repeat(2);
		Tensor tri_1_indices_col = torch::tensor({ n1_idx * 3 , n0_idx * 3 }, torch::kLong).repeat_interleave(3) + torch::arange(3, torch::kLong).repeat(2);
		Tensor tri_2_indices_col = torch::tensor({ n2_idx * 3 , n0_idx * 3 }, torch::kLong).repeat_interleave(3) + torch::arange(3, torch::kLong).repeat(2);

		Tensor tri_1_spa_indices = torch::stack({ tri_indices_row, tri_1_indices_col }, 0);
		Tensor tri_2_spa_indices = torch::stack({ tri_indices_row, tri_2_indices_col }, 0);

		auto tri_1_indices = torch::sparse_coo_tensor(tri_1_spa_indices, tri_values, { 3, num_nodes * 3 }, torch::kInt8);
		auto tri_2_indices = torch::sparse_coo_tensor(tri_2_spa_indices, tri_values, { 3, num_nodes * 3 }, torch::kInt8);

		tris_indices_1.push_back(tri_1_indices);
		tris_indices_2.push_back(tri_2_indices);

		faces_invDm.push_back(cloths[0].mesh.faces[f]->invDm);

		faces_m.push_back(face->m_a);

		Tensor d = face->invDm.flatten();
		Tensor Du = torch::cat({ (-d[0] - d[2]) * EYE3, d[0] * EYE3, d[2] * EYE3 }, 1);
		Tensor Dv = torch::cat({ (-d[1] - d[3]) * EYE3, d[1] * EYE3, d[3] * EYE3 }, 1);

		Dus.push_back(Du);
		Dvs.push_back(Dv);

		Tensor stretch_f_tensor = torch::zeros({ num_nodes * 3, 9 }, torch::kInt8);

		stretch_f_tensor.index_put_({ Slice(n0_idx * 3, n0_idx * 3 + 3), Slice(0, 3) }, torch::eye(3, torch::kInt8));
		stretch_f_tensor.index_put_({ Slice(n1_idx * 3, n1_idx * 3 + 3), Slice(3, 6) }, torch::eye(3, torch::kInt8));
		stretch_f_tensor.index_put_({ Slice(n2_idx * 3, n2_idx * 3 + 3), Slice(6, 9) }, torch::eye(3, torch::kInt8));

		int d_n0_x_d_n0 = n0_idx * (num_nodes * 3 * 3) + n0_idx * 3;
		int d_n0_x_d_n1 = n0_idx * (num_nodes * 3 * 3) + n1_idx * 3;
		int d_n0_x_d_n2 = n0_idx * (num_nodes * 3 * 3) + n2_idx * 3;

		int d_n0_y_d_n0 = n0_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n0_idx * 3;
		int d_n0_y_d_n1 = n0_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n1_idx * 3;
		int d_n0_y_d_n2 = n0_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n2_idx * 3;

		int d_n0_z_d_n0 = n0_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n0_idx * 3;
		int d_n0_z_d_n1 = n0_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n1_idx * 3;
		int d_n0_z_d_n2 = n0_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n2_idx * 3;

		int d_n1_x_d_n0 = n1_idx * (num_nodes * 3 * 3) + n0_idx * 3;
		int d_n1_x_d_n1 = n1_idx * (num_nodes * 3 * 3) + n1_idx * 3;
		int d_n1_x_d_n2 = n1_idx * (num_nodes * 3 * 3) + n2_idx * 3;

		int d_n1_y_d_n0 = n1_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n0_idx * 3;
		int d_n1_y_d_n1 = n1_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n1_idx * 3;
		int d_n1_y_d_n2 = n1_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n2_idx * 3;

		int d_n1_z_d_n0 = n1_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n0_idx * 3;
		int d_n1_z_d_n1 = n1_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n1_idx * 3;
		int d_n1_z_d_n2 = n1_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n2_idx * 3;

		int d_n2_x_d_n0 = n2_idx * (num_nodes * 3 * 3) + n0_idx * 3;
		int d_n2_x_d_n1 = n2_idx * (num_nodes * 3 * 3) + n1_idx * 3;
		int d_n2_x_d_n2 = n2_idx * (num_nodes * 3 * 3) + n2_idx * 3;

		int d_n2_y_d_n0 = n2_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n0_idx * 3;
		int d_n2_y_d_n1 = n2_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n1_idx * 3;
		int d_n2_y_d_n2 = n2_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n2_idx * 3;

		int d_n2_z_d_n0 = n2_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n0_idx * 3;
		int d_n2_z_d_n1 = n2_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n1_idx * 3;
		int d_n2_z_d_n2 = n2_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n2_idx * 3;

		Tensor row_index = torch::tensor({
			d_n0_x_d_n0, d_n0_x_d_n1, d_n0_x_d_n2, d_n0_y_d_n0, d_n0_y_d_n1, d_n0_y_d_n2, d_n0_z_d_n0, d_n0_z_d_n1, d_n0_z_d_n2,
			d_n1_x_d_n0, d_n1_x_d_n1, d_n1_x_d_n2, d_n1_y_d_n0, d_n1_y_d_n1, d_n1_y_d_n2, d_n1_z_d_n0, d_n1_z_d_n1, d_n1_z_d_n2,
			d_n2_x_d_n0, d_n2_x_d_n1, d_n2_x_d_n2, d_n2_y_d_n0, d_n2_y_d_n1, d_n2_y_d_n2, d_n2_z_d_n0, d_n2_z_d_n1, d_n2_z_d_n2
			}, torch::kLong);

		row_index = row_index.repeat_interleave(3) + torch::tensor({ 0, 1, 2 }, torch::kInt).repeat(27);

		Tensor col_index = torch::arange(81, torch::kLong);

		Tensor indices = torch::stack({ row_index, col_index }, 0);

		Tensor values = torch::ones(81, torch::kInt8);

		auto stretch_J_tensor = torch::sparse_coo_tensor(indices, values, { num_nodes * 3 * num_nodes * 3, 3 * 3 * 3 * 3 }, torch::kInt8);

		stretch_f_tensors.push_back(stretch_f_tensor.to_sparse());
		stretch_J_tensors.push_back(stretch_J_tensor);

	}

	for (int e = 0; e < num_edges; ++e)
	{
		if (e % 100 == 0) {
			std::cout << "Processing Edge: " << e << "/" << num_edges << std::endl;
		}

		const Edge* edge = cloths[0].mesh.edges[e];

		// Rule out the egdes that are not between to faces
		if (edge->adj_faces[0] == nullptr || edge->adj_faces[1] == nullptr)
			continue;

		int n0_idx = edge->nodes[0]->index;
		int n1_idx = edge->nodes[1]->index;
		int n2_idx = edge_opp_vert(edge, 0)->node->index;
		int n3_idx = edge_opp_vert(edge, 1)->node->index;

		edges_l.push_back(edge->l);
		edges_ldaa.push_back(edge->ldaa);
		edges_bang.push_back(edge->bias_angle);

		theta_ideals.push_back(edge->theta_ideal);

		Tensor edge_values = torch::tensor({ 1, 1, 1, -1, -1, -1 }, torch::kInt8);

		Tensor edge_indices_row = torch::arange(3, torch::kLong).repeat(2);
		Tensor edge_indices_col = torch::tensor({ n0_idx * 3 , n1_idx * 3 }, torch::kLong).repeat_interleave(3) + torch::arange(3, torch::kLong).repeat(2);

		Tensor edge_spa_indices = torch::stack({ edge_indices_row, edge_indices_col }, 0);

		auto edge_indices = torch::sparse_coo_tensor(edge_spa_indices, edge_values, { 3, num_nodes * 3 }, torch::kInt8);

		edges_indices.push_back(edge_indices);

		int f1_idx = edge->adj_faces[0]->index;
		int f2_idx = edge->adj_faces[1]->index;

		edges_adf_a.push_back(edge->adj_faces[0]->m_a + edge->adj_faces[1]->m_a);

		Tensor node_pos = torch::tensor({ 1, 1, 1 }, torch::kInt8);

		Tensor node_row_idx = torch::arange(3, torch::kLong);

		Tensor edge_ajf_indices_col_1 = torch::tensor({ f1_idx * 3, f1_idx * 3, f1_idx * 3 }, torch::kLong) + torch::arange(3, torch::kLong);
		Tensor edge_ajf_indices_col_2 = torch::tensor({ f2_idx * 3, f2_idx * 3, f2_idx * 3 }, torch::kLong) + torch::arange(3, torch::kLong);

		Tensor edge_ajf_spa_indices_1 = torch::stack({ node_row_idx, edge_ajf_indices_col_1 }, 0);
		Tensor edge_ajf_spa_indices_2 = torch::stack({ node_row_idx, edge_ajf_indices_col_2 }, 0);

		auto edge_ajf_indices_1 = torch::sparse_coo_tensor(edge_ajf_spa_indices_1, node_pos, { 3, num_faces * 3 }, torch::kInt8);
		auto edge_ajf_indices_2 = torch::sparse_coo_tensor(edge_ajf_spa_indices_2, node_pos, { 3, num_faces * 3 }, torch::kInt8);

		edges_ajf_indices_1.push_back(edge_ajf_indices_1);
		edges_ajf_indices_2.push_back(edge_ajf_indices_2);

		Tensor n0_indices_col = torch::tensor({ n0_idx * 3, n0_idx * 3 + 1, n0_idx * 3 + 2 }, torch::kLong);
		Tensor n1_indices_col = torch::tensor({ n1_idx * 3, n1_idx * 3 + 1, n1_idx * 3 + 2 }, torch::kLong);
		Tensor n2_indices_col = torch::tensor({ n2_idx * 3, n2_idx * 3 + 1, n2_idx * 3 + 2 }, torch::kLong);
		Tensor n3_indices_col = torch::tensor({ n3_idx * 3, n3_idx * 3 + 1, n3_idx * 3 + 2 }, torch::kLong);

		Tensor n0_spa_idx = torch::stack({ node_row_idx, n0_indices_col }, 0);
		Tensor n1_spa_idx = torch::stack({ node_row_idx, n1_indices_col }, 0);
		Tensor n2_spa_idx = torch::stack({ node_row_idx, n2_indices_col }, 0);
		Tensor n3_spa_idx = torch::stack({ node_row_idx, n3_indices_col }, 0);

		auto n0_indices_t = torch::sparse_coo_tensor(n0_spa_idx, node_pos, { 3, num_nodes * 3 }, torch::kInt8);
		auto n1_indices_t = torch::sparse_coo_tensor(n1_spa_idx, node_pos, { 3, num_nodes * 3 }, torch::kInt8);
		auto n2_indices_t = torch::sparse_coo_tensor(n2_spa_idx, node_pos, { 3, num_nodes * 3 }, torch::kInt8);
		auto n3_indices_t = torch::sparse_coo_tensor(n3_spa_idx, node_pos, { 3, num_nodes * 3 }, torch::kInt8);

		bendings_nodes_0.push_back(n0_indices_t);
		bendings_nodes_1.push_back(n1_indices_t);
		bendings_nodes_2.push_back(n2_indices_t);
		bendings_nodes_3.push_back(n3_indices_t);

		Tensor bending_f_tensor = torch::zeros({ num_nodes * 3, 12 }, torch::kInt8);

		bending_f_tensor.index_put_({ Slice(n0_idx * 3, n0_idx * 3 + 3), Slice(0, 3) }, torch::eye(3, torch::kInt8));
		bending_f_tensor.index_put_({ Slice(n1_idx * 3, n1_idx * 3 + 3), Slice(3, 6) }, torch::eye(3, torch::kInt8));
		bending_f_tensor.index_put_({ Slice(n2_idx * 3, n2_idx * 3 + 3), Slice(6, 9) }, torch::eye(3, torch::kInt8));
		bending_f_tensor.index_put_({ Slice(n3_idx * 3, n3_idx * 3 + 3), Slice(9, 12) }, torch::eye(3, torch::kInt8));

		int d_n0_x_d_n0 = n0_idx * (num_nodes * 3 * 3) + n0_idx * 3;
		int d_n0_x_d_n1 = n0_idx * (num_nodes * 3 * 3) + n1_idx * 3;
		int d_n0_x_d_n2 = n0_idx * (num_nodes * 3 * 3) + n2_idx * 3;
		int d_n0_x_d_n3 = n0_idx * (num_nodes * 3 * 3) + n3_idx * 3;

		int d_n0_y_d_n0 = n0_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n0_idx * 3;
		int d_n0_y_d_n1 = n0_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n1_idx * 3;
		int d_n0_y_d_n2 = n0_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n2_idx * 3;
		int d_n0_y_d_n3 = n0_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n3_idx * 3;

		int d_n0_z_d_n0 = n0_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n0_idx * 3;
		int d_n0_z_d_n1 = n0_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n1_idx * 3;
		int d_n0_z_d_n2 = n0_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n2_idx * 3;
		int d_n0_z_d_n3 = n0_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n3_idx * 3;

		int d_n1_x_d_n0 = n1_idx * (num_nodes * 3 * 3) + n0_idx * 3;
		int d_n1_x_d_n1 = n1_idx * (num_nodes * 3 * 3) + n1_idx * 3;
		int d_n1_x_d_n2 = n1_idx * (num_nodes * 3 * 3) + n2_idx * 3;
		int d_n1_x_d_n3 = n1_idx * (num_nodes * 3 * 3) + n3_idx * 3;

		int d_n1_y_d_n0 = n1_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n0_idx * 3;
		int d_n1_y_d_n1 = n1_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n1_idx * 3;
		int d_n1_y_d_n2 = n1_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n2_idx * 3;
		int d_n1_y_d_n3 = n1_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n3_idx * 3;

		int d_n1_z_d_n0 = n1_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n0_idx * 3;
		int d_n1_z_d_n1 = n1_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n1_idx * 3;
		int d_n1_z_d_n2 = n1_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n2_idx * 3;
		int d_n1_z_d_n3 = n1_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n3_idx * 3;

		int d_n2_x_d_n0 = n2_idx * (num_nodes * 3 * 3) + n0_idx * 3;
		int d_n2_x_d_n1 = n2_idx * (num_nodes * 3 * 3) + n1_idx * 3;
		int d_n2_x_d_n2 = n2_idx * (num_nodes * 3 * 3) + n2_idx * 3;
		int d_n2_x_d_n3 = n2_idx * (num_nodes * 3 * 3) + n3_idx * 3;

		int d_n2_y_d_n0 = n2_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n0_idx * 3;
		int d_n2_y_d_n1 = n2_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n1_idx * 3;
		int d_n2_y_d_n2 = n2_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n2_idx * 3;
		int d_n2_y_d_n3 = n2_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n3_idx * 3;

		int d_n2_z_d_n0 = n2_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n0_idx * 3;
		int d_n2_z_d_n1 = n2_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n1_idx * 3;
		int d_n2_z_d_n2 = n2_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n2_idx * 3;
		int d_n2_z_d_n3 = n2_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n3_idx * 3;

		int d_n3_x_d_n0 = n3_idx * (num_nodes * 3 * 3) + n0_idx * 3;
		int d_n3_x_d_n1 = n3_idx * (num_nodes * 3 * 3) + n1_idx * 3;
		int d_n3_x_d_n2 = n3_idx * (num_nodes * 3 * 3) + n2_idx * 3;
		int d_n3_x_d_n3 = n3_idx * (num_nodes * 3 * 3) + n3_idx * 3;

		int d_n3_y_d_n0 = n3_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n0_idx * 3;
		int d_n3_y_d_n1 = n3_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n1_idx * 3;
		int d_n3_y_d_n2 = n3_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n2_idx * 3;
		int d_n3_y_d_n3 = n3_idx * (num_nodes * 3 * 3) + (num_nodes * 3) + n3_idx * 3;

		int d_n3_z_d_n0 = n3_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n0_idx * 3;
		int d_n3_z_d_n1 = n3_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n1_idx * 3;
		int d_n3_z_d_n2 = n3_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n2_idx * 3;
		int d_n3_z_d_n3 = n3_idx * (num_nodes * 3 * 3) + (num_nodes * 3) * 2 + n3_idx * 3;

		Tensor row_index = torch::tensor({
			d_n0_x_d_n0, d_n0_x_d_n1, d_n0_x_d_n2, d_n0_x_d_n3,
			d_n0_y_d_n0, d_n0_y_d_n1, d_n0_y_d_n2, d_n0_y_d_n3,
			d_n0_z_d_n0, d_n0_z_d_n1, d_n0_z_d_n2, d_n0_z_d_n3,
			d_n1_x_d_n0, d_n1_x_d_n1, d_n1_x_d_n2, d_n1_x_d_n3,
			d_n1_y_d_n0, d_n1_y_d_n1, d_n1_y_d_n2, d_n1_y_d_n3,
			d_n1_z_d_n0, d_n1_z_d_n1, d_n1_z_d_n2, d_n1_z_d_n3,
			d_n2_x_d_n0, d_n2_x_d_n1, d_n2_x_d_n2, d_n2_x_d_n3,
			d_n2_y_d_n0, d_n2_y_d_n1, d_n2_y_d_n2, d_n2_y_d_n3,
			d_n2_z_d_n0, d_n2_z_d_n1, d_n2_z_d_n2, d_n2_z_d_n3,
			d_n3_x_d_n0, d_n3_x_d_n1, d_n3_x_d_n2, d_n3_x_d_n3,
			d_n3_y_d_n0, d_n3_y_d_n1, d_n3_y_d_n2, d_n3_y_d_n3,
			d_n3_z_d_n0, d_n3_z_d_n1, d_n3_z_d_n2, d_n3_z_d_n3 }, torch::kLong);

		row_index = row_index.repeat_interleave(3) + torch::tensor({ 0, 1, 2 }, torch::kInt).repeat(48);

		Tensor col_index = torch::arange(144, torch::kLong);

		Tensor indices = torch::stack({ row_index, col_index }, 0);

		Tensor values = torch::ones(144, torch::kInt8);

		auto bending_J_tensor = torch::sparse_coo_tensor(indices, values, { num_nodes * 3 * num_nodes * 3, 4 * 3 * 4 * 3 }, torch::kInt8);

		bending_f_tensors.push_back(bending_f_tensor.to_sparse());
		bending_J_tensors.push_back(bending_J_tensor);
	}

	cloths[0].node_m_matrix = torch::from_blob(node_m.data(), { num_nodes * num_faces }, TNOPT).view({ num_nodes, num_faces }).to(torch::Device(torch::kCUDA, device_idx));

	cloths[0].face_indices_1_cuda = torch::cat(tris_indices_1, 0).to(tensor_opt);
	cloths[0].face_indices_2_cuda = torch::cat(tris_indices_2, 0).to(tensor_opt);

	cloths[0].faces_invDm_cuda = torch::stack(faces_invDm, 0).to(torch::Device(torch::kCUDA, device_idx));

	cloths[0].faces_m_cuda = torch::stack(faces_m).to(torch::Device(torch::kCUDA, device_idx));

	cloths[0].Du_cuda = torch::stack(Dus).to(torch::Device(torch::kCUDA, device_idx));
	cloths[0].Dv_cuda = torch::stack(Dvs).to(torch::Device(torch::kCUDA, device_idx));

	cloths[0].SF_T_cuda = torch::cat(stretch_f_tensors, 1).to(tensor_opt).to_sparse_csr();
	cloths[0].SJ_T_cuda = torch::cat(stretch_J_tensors, 1).to(tensor_opt).to_sparse_csr();

	cloths[0].edges_indices_t_cuda = torch::cat(edges_indices, 0).to(tensor_opt);

	cloths[0].edges_adf_a_cuda = torch::stack(edges_adf_a).to(torch::Device(torch::kCUDA, device_idx));

	cloths[0].edges_l_cuda = torch::stack(edges_l).to(torch::Device(torch::kCUDA, device_idx));

	cloths[0].edges_ldaa_cuda = torch::stack(edges_ldaa).to(torch::Device(torch::kCUDA, device_idx));

	cloths[0].edges_bang_cuda = torch::stack(edges_bang).to(torch::Device(torch::kCUDA, device_idx));

	cloths[0].theta_ideals_cuda = torch::stack(theta_ideals).to(torch::Device(torch::kCUDA, device_idx));

	cloths[0].edges_ajf_indices_1_cuda = torch::cat(edges_ajf_indices_1, 0).to(tensor_opt);
	cloths[0].edges_ajf_indices_2_cuda = torch::cat(edges_ajf_indices_2, 0).to(tensor_opt);

	cloths[0].bendings_node0_cuda = torch::cat(bendings_nodes_0, 0).to(tensor_opt);
	cloths[0].bendings_node1_cuda = torch::cat(bendings_nodes_1, 0).to(tensor_opt);
	cloths[0].bendings_node2_cuda = torch::cat(bendings_nodes_2, 0).to(tensor_opt);
	cloths[0].bendings_node3_cuda = torch::cat(bendings_nodes_3, 0).to(tensor_opt);

	cloths[0].BF_T_cuda = torch::cat(bending_f_tensors, 1).to(tensor_opt).to_sparse_csr();
	cloths[0].BJ_T_cuda = torch::cat(bending_J_tensors, 1).to(tensor_opt).to_sparse_csr();

	std::vector<int> handle_indices = std::vector<int>(num_nodes, 0);
	std::vector<Tensor> init_pos_vec;

	// Cusick's drape handles
	for (int n = 0; n < num_nodes; ++n)
	{
		auto dist = torch::norm(cloths[0].mesh.nodes[n]->x);

		init_pos_vec.push_back(cloths[0].mesh.nodes[n]->x);

		if (dist.item<double>() < 0.09) {
			handle_indices[n] = 1;
		}
	}

	cloths[0].init_pos_t_cuda = torch::cat(init_pos_vec).to(torch::Device(torch::kCUDA, device_idx));
	cloths[0].handle_indices_t_cuda = torch::diag(torch::tensor(handle_indices, tensor_opt).repeat_interleave(3)).to_sparse();
}

void Simulation::advance_step()
{
	if (dev_wordy)
		std::cout << "Advanced Step" << std::endl;

	std::cout << "In Simulation Step: " << step << std::endl;

	//get_constraints();

	physics_step();

	obstacle_step();

	collision_step();

	//stable_step();

	//delete_constraints();

	step++;
}

std::pair<Tensor, Tensor> Simulation::advance_step(const Tensor& pos, const Tensor& vel)
{
	if (dev_wordy)
		std::cout << "Advanced Step" << std::endl;

	std::cout << "In Simulation Step: " << step << std::endl;

	implicit_update(cloths[0], pos, vel, step_time);

	//obstacle_step();

	//collision_step();

	step++;

	return cloths[0].get_pos_vel();
}

void Simulation::advance_step_cuda()
{
	if (dev_wordy)
		std::cout << "Advanced Step" << std::endl;

	std::cout << "In Simulation Step: " << step << std::endl;

	physics_step_cuda();

	step++;
}

std::pair<Tensor, Tensor> Simulation::advance_step_cuda(const Tensor& pos, const Tensor& vel)
{
	if (dev_wordy)
		std::cout << "Advanced Step" << std::endl;

	std::cout << "In Simulation Step: " << step << std::endl;

	implicit_update_cuda(cloths[0], pos, vel, step_time_cuda, device_idx, tensor_opt);

	step++;

	return std::make_pair(cloths[0].pos_cuda, cloths[0].vel_cuda);

	// return cloths[0].get_pos_vel_cuda();
}

void Simulation::physics_step()
{
	if (dev_wordy)
		std::cout << "Physics Step" << std::endl;

	for (auto& cloth : cloths) {
		implicit_update(cloth, handles, step_time);
	}
}

void Simulation::physics_step_cuda()
{
	if (dev_wordy)
		std::cout << "Physics Step" << std::endl;

	for (auto& cloth : cloths) {
		implicit_update_cuda(cloth, step_time, step_time_cuda);
	}
}

void Simulation::obstacle_step()
{
	if (dev_wordy)
		std::cout << "Obstacle falling step" << std::endl;

	for (int o = 0; o < obstacles.size(); ++o)
	{
		for (int n = 0; n < obstacles[o].mesh.nodes.size(); ++n)
		{
			obstacles[o].mesh.nodes[n]->x = obstacles[o].mesh.nodes[n]->x + step_time * falling_velocity;
		}

		//obstacles[o].mesh.compute_ms_data();
		//obstacles[o].mesh.compute_ws_data();
	}
}

void Simulation::stable_step()
{
	for (int c = 0; c < cloths.size(); ++c)
	{
		for (int n = 0; n < cloths[c].mesh.nodes.size(); ++n)
		{
			cloths[c].mesh.nodes[n]->v = torch::zeros({ 3 }, TNOPT);
		}
	}
}

void Simulation::get_constraints()
{
	//std::cout << "Get Handle Constraints" << std::endl;

	for (int h = 0; h < handles.size(); ++h)
	{
		std::vector<Constraint*> cons_temp = handles[h]->get_constraints(step * step_time);
		cons.insert(cons.end(), cons_temp.begin(), cons_temp.end());
	}

	// Proximity Constraints
	
	torch::Tensor mu = torch::tensor({ 0.5 }, TNOPT);
	torch::Tensor mu_obs = torch::tensor({ 0.7 }, TNOPT);

	//Proximity prox;

	std::vector<Mesh*> cloth_mesh;
	std::vector<Mesh*> obs_mesh;

	for (int c = 0; c < cloths.size(); ++c)
	{
		cloth_mesh.push_back(&cloths[c].mesh);
	}

	for (int o = 0; o < obstacles.size(); ++o)
	{
		obs_mesh.push_back(&obstacles[o].mesh);
	}

	std::vector<Constraint*> cons_proximity_temp = prox.proximity_constraints(cloth_mesh, obs_mesh, mu, mu_obs);

	cons.insert(cons.end(), cons_proximity_temp.begin(), cons_proximity_temp.end());
}

void Simulation::delete_constraints()
{
	if (dev_wordy)
		std::cout << "Delete Constraints" << std::endl;

	for (int c = 0; c < cons.size(); ++c)
	{
		delete cons[c];
	}

	cons.clear();
}

void Simulation::collision_step()
{
	// std::cout << "Collision Step" << std::endl;

	std::vector<Tensor> x_old;
	
	for (int c = 0; c < cloths.size(); ++c)
		for (int n = 0; n < cloths[c].mesh.nodes.size(); ++n)
		{
			x_old.push_back(cloths[c].mesh.nodes[n]->x);
		}

	std::vector<Mesh*> cloth_mesh;
	std::vector<Mesh*> obs_mesh;

	for (int c = 0; c < cloths.size(); ++c) {
		cloth_mesh.push_back(&cloths[c].mesh);
	}

	for (int o = 0; o < obstacles.size(); ++o) {
		obs_mesh.push_back(&obstacles[o].mesh);
	}

	Collision collision;

	collision.collision_response(cloth_mesh, obs_mesh);

	// Update Cloth Velocity after Collision
	for (int c = 0; c < cloths.size(); ++c)
		for (int n = 0; n < x_old.size(); ++n)
		{
			Node* node = cloths[c].mesh.nodes[n];
			node->x0 = node->x;
			node->v = node->v + (node->x - x_old[n]) / step_time;
		}

	for (int o = 0; o < obstacles.size(); o++)
	{
		obstacles[o].mesh.update_x0();
	}
}