#include "Physics.h"
#include "DDE.h"
#include "Geometry.h"

#include <cmath>
#include <chrono>

using namespace std::chrono;
using namespace torch::indexing;

std::vector<int> indices(const Node* n0, const Node* n1, const Node* n2)
{
	std::vector<int> ix(3);
	ix[0] = n0->index;
	ix[1] = n1->index;
	ix[2] = n2->index;
	return ix;
}

std::vector<int> indices(const Node* n0, const Node* n1, const Node* n2, const Node* n3)
{
	std::vector<int> ix(4);
	ix[0] = n0->index;
	ix[1] = n1->index;
	ix[2] = n2->index;
	ix[3] = n3->index;
	return ix;
}

Tensor wind_force(const Face* face, const Wind& wind)
{
	// Face Velocity: average of three nodes' velocity
	Tensor face_velocity = (face->v[0]->node->v + face->v[1]->node->v + face->v[2]->node->v) / 3.0;
	// Relative velocity between face and wind
	Tensor relative_velocity = wind.velocity - face_velocity;
	// Relative Velocity in the normal direction of the face
	Tensor velocity_normal = torch::dot(face->n, relative_velocity);
	// Relative Velocity in the tangent direction of the face
	Tensor velocity_tangent = relative_velocity - velocity_normal * face->n;
	// Wind force on the face
	return wind.density * face->m_a * torch::abs(velocity_normal) * velocity_normal * face->n +
		wind.drag * face->m_a * velocity_tangent;
}

void add_noise_forces(const Cloth& cloth, const Tensor& noise_force, Tensor& fext, Tensor& Jext)
{
	// Add noise force on the free nodes
	for (int n = 0; n < cloth.mesh.nodes.size(); ++n)
		if (n >= 51)
			fext[n] = fext[n] + noise_force[n - 51];
}

void add_external_forces(const Cloth& cloth, const Tensor& gravity, const Wind& wind, std::vector<Tensor>& fext, std::vector<Tensor>& Jext)
{
	//std::cout << "In Function Add Gravity and Wind Force" << std::endl;

	for (int n = 0; n < cloth.mesh.nodes.size(); ++n)
	{
		fext[n] = fext[n] + cloth.mesh.nodes[n]->m * gravity;
	}

	for (int f = 0; f < cloth.mesh.faces.size(); ++f)
	{
		const Face* face = cloth.mesh.faces[f];
		Tensor wf = wind_force(face, wind);
		for (int v = 0; v < 3; ++v)
			fext[face->v[v]->node->index] = fext[face->v[v]->node->index] + (wf / 3.0);
	}
}

std::pair<Tensor, Tensor> batch_stretching_force(const Tensor& batch_F, const Tensor& batch_stret, const Tensor& batch_du, const Tensor& batch_dv, const Tensor& batch_a)
{	
	
	Tensor G = (torch::bmm(batch_F.permute({ 0, 2, 1 }), batch_F) - torch::eye(2, TNOPT).unsqueeze(0)).reshape({ -1, 4 }) * 0.5;
	G = G.t();
	
	Tensor k = batch_stretching_stiffness(G, batch_stret);

	const Tensor& xu = batch_F.slice(2, 0, 1);
	const Tensor& xv = batch_F.slice(2, 1, 2);

	Tensor Dut = batch_du.permute({ 0, 2, 1 });
	Tensor Dvt = batch_dv.permute({ 0, 2, 1 });

	Tensor fuu = torch::bmm(Dut, xu).squeeze();
	Tensor fvv = torch::bmm(Dvt, xv).squeeze();
	Tensor fuv = (torch::bmm(Dut, xv) + torch::bmm(Dvt, xu)).squeeze();

	Tensor grad_e = k[0] * G[0] * fuu.t() + k[2] * G[3] * fvv.t() +
		k[1] * (G[0] * fvv.t() + G[3] * fuu.t()) + k[3] * G[1] * fuv.t();

	G = torch::relu(G);

	Tensor DutDu = torch::bmm(Dut, batch_du).permute({1, 2, 0});
	Tensor DvtDv = torch::bmm(Dvt, batch_dv).permute({1, 2, 0});

	Tensor hess_e = k[0] * (bmm(fuu.unsqueeze(2), fuu.unsqueeze(1)).permute({ 1,2,0 }) + G[0] * DutDu)
			+ k[2] * (bmm(fvv.unsqueeze(2), fvv.unsqueeze(1)).permute({ 1,2,0 }) + G[3] * DvtDv)
			+ k[1] * (bmm(fuu.unsqueeze(2), fvv.unsqueeze(1)).permute({ 1,2,0 }) + G[0] * DvtDv
			+ bmm(fvv.unsqueeze(2), fuu.unsqueeze(1)).permute({ 1,2,0 }) + G[3] * DutDu)
			+ 0.5 * k[3] * (bmm(fuv.unsqueeze(2), fuv.unsqueeze(1)).permute({ 1,2,0 }));

	return std::make_pair(-(batch_a * hess_e).permute({ 2,0,1 }), -(batch_a * grad_e).t());
}

std::pair<Tensor, Tensor> batch_bending_force(const std::vector<Tensor>& bat_x, const Tensor& bat_a, const Tensor& bat_theta, const Tensor& bat_n, const Tensor& bat_bend, const Tensor& bat_ldaa, const Tensor& bat_bang, const Tensor& bat_theta_ideal, const Tensor& bat_oritheta)
{

	Tensor e = bat_x[1] - bat_x[0];
	Tensor dote = torch::sum(e * e, 1);
	Tensor norme = 1e-3 * torch::sqrt(dote);

	Tensor x2mx0 = bat_x[2] - bat_x[0];
	Tensor x3mx0 = bat_x[3] - bat_x[0];

	Tensor t0 = torch::sum(e * (x2mx0), 1) / dote;
	Tensor t1 = torch::sum(e * (x3mx0), 1) / dote;
	Tensor h0 = torch::max(torch::norm(x2mx0 - e * t0.unsqueeze(1), 2, { 1 }), norme);
	Tensor h1 = torch::max(torch::norm(x3mx0 - e * t1.unsqueeze(1), 2, { 1 }), norme);

	Tensor n0 = bat_n[0] / h0.unsqueeze(1);
	Tensor n1 = bat_n[1] / h1.unsqueeze(1);

	Tensor w_f = torch::stack({ t0 - 1, t1 - 1, -t0, -t1 }).t().reshape({ -1,2,2 });
	Tensor dtheta = torch::bmm(w_f, torch::stack({ n0, n1 }, 1)).squeeze(); //nx2x3
	dtheta = torch::cat({ dtheta, n0.unsqueeze(1), n1.unsqueeze(1) }, 1).reshape({ -1,12 }); //nx12

	Tensor ke = batch_bending_stiffness(bat_oritheta * bat_ldaa * 0.05, bat_bang, bat_bend);

	ke = ke * (-(bat_ldaa * bat_ldaa) * bat_a / 4);

	return std::make_pair(ke.unsqueeze(1).unsqueeze(1) * bmm(dtheta.unsqueeze(2), dtheta.unsqueeze(1)),
		(ke * (bat_theta - bat_theta_ideal)).unsqueeze(1) * dtheta);
}

std::pair<Tensor, Tensor> batch_stretching_force_cuda(const Tensor& batch_F, const Tensor& batch_stret, const Tensor& batch_du, const Tensor& batch_dv, const Tensor& batch_a, const torch::TensorOptions& tensor_opt)
{
	Tensor G = (torch::bmm(batch_F.permute({ 0, 2, 1 }), batch_F) - torch::eye(2, tensor_opt).unsqueeze(0)).reshape({ -1, 4 }) * 0.5;
	G = G.t();

	Tensor k = batch_stretching_stiffness_cuda(G, batch_stret);

	const Tensor& xu = batch_F.slice(2, 0, 1);
	const Tensor& xv = batch_F.slice(2, 1, 2);

	Tensor Dut = batch_du.permute({ 0, 2, 1 });
	Tensor Dvt = batch_dv.permute({ 0, 2, 1 });

	Tensor fuu = torch::bmm(Dut, xu).squeeze();
	Tensor fvv = torch::bmm(Dvt, xv).squeeze();
	Tensor fuv = (torch::bmm(Dut, xv) + torch::bmm(Dvt, xu)).squeeze();

	Tensor grad_e = k[0] * G[0] * fuu.t() + k[2] * G[3] * fvv.t() +
		k[1] * (G[0] * fvv.t() + G[3] * fuu.t()) + k[3] * G[1] * fuv.t();

	G = torch::relu(G);

	Tensor DutDu = torch::bmm(Dut, batch_du).permute({ 1, 2, 0 });
	Tensor DvtDv = torch::bmm(Dvt, batch_dv).permute({ 1, 2, 0 });

	Tensor hess_e = k[0] * (bmm(fuu.unsqueeze(2), fuu.unsqueeze(1)).permute({ 1,2,0 }) + G[0] * DutDu)
		+ k[2] * (bmm(fvv.unsqueeze(2), fvv.unsqueeze(1)).permute({ 1,2,0 }) + G[3] * DvtDv)
		+ k[1] * (bmm(fuu.unsqueeze(2), fvv.unsqueeze(1)).permute({ 1,2,0 }) + G[0] * DvtDv
			+ bmm(fvv.unsqueeze(2), fuu.unsqueeze(1)).permute({ 1,2,0 }) + G[3] * DutDu)
		+ 0.5 * k[3] * (bmm(fuv.unsqueeze(2), fuv.unsqueeze(1)).permute({ 1,2,0 }));

	return std::make_pair(-(batch_a * hess_e).permute({ 2,0,1 }), -(batch_a * grad_e).t());
}

std::pair<Tensor, Tensor> batch_bending_force_cuda(const std::vector<Tensor>& bat_x, const Tensor& bat_a, const Tensor& bat_theta, const Tensor& bat_n, const Tensor& bat_bend, const Tensor& bat_ldaa, const Tensor& bat_bang, const Tensor& bat_theta_ideal, const Tensor& bat_oritheta)
{
	Tensor e = bat_x[1] - bat_x[0];					// Bending Edge
	Tensor dote = torch::sum(e * e, 1);				// Dot Produce of Bending Edge Vector
	Tensor norme = 1e-3 * torch::sqrt(dote);		// Length of Bending Edge Vector

	Tensor x2mx0 = bat_x[2] - bat_x[0];		// x2 minus x0
	Tensor x3mx0 = bat_x[3] - bat_x[0];		// x3 minus x0

	Tensor t0 = torch::sum(e * (x2mx0), 1) / dote;
	Tensor t1 = torch::sum(e * (x3mx0), 1) / dote;
	Tensor h0 = torch::max(torch::norm(x2mx0 - e * t0.unsqueeze(1), 2, { 1 }), norme);	// used to compute derivative of theta
	Tensor h1 = torch::max(torch::norm(x3mx0 - e * t1.unsqueeze(1), 2, { 1 }), norme);  // used to compute derivative of theta

	Tensor n0 = bat_n[0] / h0.unsqueeze(1);
	Tensor n1 = bat_n[1] / h1.unsqueeze(1);

	Tensor w_f = torch::stack({ t0 - 1, t1 - 1, -t0, -t1 }).t().reshape({ -1,2,2 });
	Tensor dtheta = torch::bmm(w_f, torch::stack({ n0, n1 }, 1)).squeeze(); //nx2x3
	dtheta = torch::cat({ dtheta, n0.unsqueeze(1), n1.unsqueeze(1) }, 1).reshape({ -1,12 }); //nx12

	Tensor ke = batch_bending_stiffness_cuda(bat_oritheta * bat_ldaa * 0.05, bat_bang, bat_bend);

	ke = ke * (-(bat_ldaa * bat_ldaa) * bat_a / 4);

	return std::make_pair(ke.unsqueeze(1).unsqueeze(1) * bmm(dtheta.unsqueeze(2), dtheta.unsqueeze(1)), (ke * (bat_theta - bat_theta_ideal)).unsqueeze(1) * dtheta);
}

Tensor add_internal_forces(const Cloth& cloth, const Tensor& ns_t, Tensor& Force)
{
	auto num_face = static_cast<long long>(cloth.mesh.faces.size());

	auto Dm_w_1 = torch::mv(cloth.face_indices_1, ns_t);

	auto Dm_w_2 = torch::mv(cloth.face_indices_2, ns_t);

	auto Dm_w_l = torch::stack({ Dm_w_1,  Dm_w_2 }, 1);

	auto Dm_w_t = Dm_w_l.view({ num_face, 3, 2 });

	auto F_s = torch::bmm(Dm_w_t, cloth.faces_invDm);

	auto stretching = evaluate_stretching_samples(cloth.stretch_ori, cloth.grid);	// For working with checkpoint

	auto memF = batch_stretching_force(F_s, stretching, cloth.Du, cloth.Dv, cloth.faces_m);

	Force = Force + torch::mv(cloth.SF_T, memF.second.flatten());
	Tensor Jaco = torch::mv(cloth.SJ_T, memF.first.flatten()).view({ Force.size(0), Force.size(0) });

	std::vector<Tensor> bendings_nodes = std::vector<Tensor>{ 
		torch::mv(cloth.bendings_node0, ns_t).view({-1, 3}), 
		torch::mv(cloth.bendings_node1, ns_t).view({-1, 3}),
		torch::mv(cloth.bendings_node2, ns_t).view({-1, 3}),
		torch::mv(cloth.bendings_node3, ns_t).view({-1, 3}),
	};

	auto edge_length_ws = torch::mv(cloth.edges_indices_t, ns_t);

	auto edge_length_ws_block = edge_length_ws.view({ -1, 3 });

	auto edge_length_ws_normal = torch::nn::functional::normalize(edge_length_ws_block, torch::nn::functional::NormalizeFuncOptions().dim(1));

	auto faces_normal = torch::cross(Dm_w_1.view({ -1, 3 }), Dm_w_2.view({ -1, 3 }));

	auto faces_normal_u = torch::nn::functional::normalize(faces_normal, torch::nn::functional::NormalizeFuncOptions().dim(1));

	auto edge_adj_face_1_normal = torch::mv(cloth.edges_ajf_indices_1, faces_normal_u.flatten()).view({ -1, 3 });
	auto edge_adj_face_2_normal = torch::mv(cloth.edges_ajf_indices_2, faces_normal_u.flatten()).view({ -1, 3 });

	auto edge_adj_faces_normals = torch::stack({ edge_adj_face_1_normal, edge_adj_face_2_normal }, 0);

	auto edge_cos = torch::bmm(edge_adj_face_1_normal.unsqueeze(1), edge_adj_face_2_normal.unsqueeze(2)).squeeze();

	auto edge_sin = torch::bmm(edge_length_ws_normal.unsqueeze(1), torch::cross(edge_adj_face_1_normal, edge_adj_face_2_normal).unsqueeze(2)).squeeze();

	auto thetas = torch::atan2(edge_sin, edge_cos);

	auto bendF = batch_bending_force(bendings_nodes, cloth.edges_adf_a, thetas, edge_adj_faces_normals,
		cloth.bending, cloth.edges_ldaa, cloth.edges_bang, cloth.theta_ideals, thetas);

	Force = Force + torch::mv(cloth.BF_T, bendF.second.flatten());
	Jaco = Jaco + torch::mv(cloth.BJ_T, bendF.first.flatten()).view({ Force.size(0), Force.size(0) });

	return Jaco;
}

Tensor add_internal_forces_cuda(const Cloth& cloth, const Tensor& ns_t, Tensor& Force, const torch::TensorOptions& tensor_opt)
{
	auto compute_force_start = high_resolution_clock::now();

	auto num_face = static_cast<long long>(cloth.mesh.faces.size());

	auto Dm_w_1 = torch::mv(cloth.face_indices_1_cuda, ns_t);

	auto Dm_w_2 = torch::mv(cloth.face_indices_2_cuda, ns_t);

	auto Dm_w_l = torch::stack({ Dm_w_1,  Dm_w_2 }, 1);

	auto Dm_w_t = Dm_w_l.view({ num_face, 3, 2 });

	auto F_s = torch::bmm(Dm_w_t, cloth.faces_invDm_cuda);

	auto stretch_ori_cuda = torch::stack({
		torch::stack(cloth.c11_cuda), 
		torch::stack(cloth.c12_cuda), 
		torch::stack(cloth.c22_cuda), 
		torch::stack(cloth.c33_cuda)}, 2);

	auto stretching_cuda = evaluate_stretching_samples(stretch_ori_cuda, cloth.grid_cuda);
	
	auto memF = batch_stretching_force_cuda(F_s, stretching_cuda, cloth.Du_cuda, cloth.Dv_cuda, cloth.faces_m_cuda, tensor_opt);

	std::vector<Tensor> bendings_nodes = std::vector<Tensor>{
		torch::mv(cloth.bendings_node0_cuda, ns_t).view({-1, 3}),
		torch::mv(cloth.bendings_node1_cuda, ns_t).view({-1, 3}),
		torch::mv(cloth.bendings_node2_cuda, ns_t).view({-1, 3}),
		torch::mv(cloth.bendings_node3_cuda, ns_t).view({-1, 3}) };

	auto edge_length_ws_block = torch::mv(cloth.edges_indices_t_cuda, ns_t).view({ -1, 3 });

	auto edge_length_ws_normal = torch::nn::functional::normalize(edge_length_ws_block, torch::nn::functional::NormalizeFuncOptions().dim(1));

	auto faces_normal = torch::cross(Dm_w_1.view({ -1, 3 }), Dm_w_2.view({ -1, 3 }));

	auto faces_normal_u = torch::nn::functional::normalize(faces_normal, torch::nn::functional::NormalizeFuncOptions().dim(1));

	auto edge_adj_face_1_normal = torch::mv(cloth.edges_ajf_indices_1_cuda, faces_normal_u.flatten()).view({ -1, 3 });
	auto edge_adj_face_2_normal = torch::mv(cloth.edges_ajf_indices_2_cuda, faces_normal_u.flatten()).view({ -1, 3 });

	auto edge_adj_faces_normals = torch::stack({ edge_adj_face_1_normal, edge_adj_face_2_normal }, 0);

	auto edge_cos = torch::bmm(edge_adj_face_1_normal.unsqueeze(1), edge_adj_face_2_normal.unsqueeze(2)).squeeze();

	auto edge_sin = torch::bmm(edge_length_ws_normal.unsqueeze(1), torch::cross(edge_adj_face_1_normal, edge_adj_face_2_normal).unsqueeze(2)).squeeze();

	auto thetas = torch::atan2(edge_sin, edge_cos);

	auto bendF = batch_bending_force_cuda(bendings_nodes, cloth.edges_adf_a_cuda, thetas, edge_adj_faces_normals,
		cloth.bending_cuda, cloth.edges_ldaa_cuda, cloth.edges_bang_cuda, cloth.theta_ideals_cuda, thetas);

	Force += (torch::mv(cloth.SF_T_cuda, memF.second.flatten()) + torch::mv(cloth.BF_T_cuda, bendF.second.flatten()));

	Tensor Jaco = torch::mv(cloth.SJ_T_cuda, memF.first.flatten()) + torch::mv(cloth.BJ_T_cuda, bendF.first.flatten());

	return Jaco.view({ Force.size(0), Force.size(0) });
}

bool contains(const Mesh& mesh, const Node* node)
{
	return node->index < mesh.nodes.size() && mesh.nodes[node->index] == node;
}

void add_constraint_forces(const Cloth& cloth, const std::vector<Constraint*>& cons, SparseMatrix& A, std::vector<Tensor>& b, Tensor dt)
{
	//std::cout << "Add Constraint Force " << std::endl;
	
	const Mesh& mesh = cloth.mesh;

	for (int c = 0; c < cons.size(); c++) 
	{		
		Tensor value = cons[c]->value();
		Tensor g = cons[c]->energy_grad(value);
		Tensor h = cons[c]->energy_hess(value);
		MeshGrad grad = cons[c]->gradient();
		// f = -g*grad
		// J = -h*ger(grad,grad)
		Tensor v_dot_grad = ZERO;

		for (MeshGrad::iterator it = grad.begin(); it != grad.end(); it++) 
		{
			const Node* node = it->first;
			v_dot_grad = v_dot_grad + dot(it->second, node->v);
		}

		for (MeshGrad::iterator it = grad.begin(); it != grad.end(); it++) 
		{
			const Node* nodei = it->first;

			if (!contains(mesh, nodei))
			{
				continue;
			}
			int ni = nodei->index;
			for (MeshGrad::iterator jt = grad.begin(); jt != grad.end(); jt++) 
			{
				const Node* nodej = jt->first;
				if (!contains(mesh, nodej))
				{
					continue;
				}
				int nj = nodej->index;
				if (dt.item<double>() == 0.0)
				{
					add_submat(A, ni, nj, h * ger(it->second, jt->second));
				}
				else
				{
					//add_submat(A, ni, nj, dt * dt * h * ger(it->second, jt->second));
					add_submat(A, ni, nj, -h * ger(it->second, jt->second));

					std::cout << "ni: " << ni << std::endl;
					std::cout << "nj: " << nj << std::endl;
				}
			}
			if (dt.item<double>() == 0.0)
			{
				b[ni] = b[ni] - g * it->second;
			}
			else
			{
				//b[ni] = b[ni] - dt * (g + dt * h * v_dot_grad) * it->second;
				b[ni] = b[ni] - g * it->second;
			}
		}
	}
}

void add_friction_forces(const Cloth& cloth, const std::vector<Constraint*> cons, SparseMatrix& A, Tensor& b, Tensor dt) 
{
	const Mesh& mesh = cloth.mesh;

	for (int c = 0; c < cons.size(); c++) 
	{
		MeshHess jac;
		MeshGrad force = cons[c]->friction(dt, jac);

		for (MeshGrad::iterator it = force.begin(); it != force.end(); it++) 
		{
			const Node* node = it->first;
			if (!contains(mesh, node))
				continue;
			b[node->index] = b[node->index] + dt * it->second;
		}

		for (MeshHess::iterator it = jac.begin(); it != jac.end(); it++) 
		{
			const Node* nodei = it->first.first, * nodej = it->first.second;
			if (!contains(mesh, nodei) || !contains(mesh, nodej))
				continue;
			add_submat(A, nodei->index, nodej->index, -dt * it->second);
		}
	}
}

void implicit_update(Cloth& cloth, std::vector<Handle*> handles, const Tensor& dt)
{
	int num_nodes = cloth.mesh.nodes.size();

	auto Force = torch::mv(cloth.M, cloth.nodes_gravity);

	std::vector<Tensor> ns, vs;

	for (int n = 0; n < num_nodes; ++n)
	{
		const Node* node = cloth.mesh.nodes[n];

		ns.push_back(cloth.mesh.nodes[n]->x);

		vs.push_back(cloth.mesh.nodes[n]->v);
	}

	Tensor ns_t = torch::cat(ns);

	Tensor vs_t = torch::cat(vs);

	Tensor Jaco = add_internal_forces(cloth, ns_t, Force);

	Force = Force - ::magic.handle_stiffness * torch::mv(cloth.handle_indices_t, ns_t - cloth.init_pos_t);

	Jaco = Jaco - ::magic.handle_stiffness * cloth.handle_indices_t;

	Tensor dv = EigenSolver::apply(cloth.M - dt * dt * Jaco, dt* (Force + dt * torch::mv(Jaco, vs_t))).reshape({ num_nodes, 3 });

	for (int n = 0; n < num_nodes; ++n)
	{
		Node* node = cloth.mesh.nodes[n];

		node->v = node->v + dv[n];
		node->x = node->x + node->v * dt;
		node->acceleration = dv[n] / dt;
	}
}

void implicit_update(Cloth& cloth, const Tensor& ns_t, const Tensor& vs_t, const Tensor& dt)
{
	int num_nodes{ static_cast<int>(cloth.mesh.nodes.size()) };
	
	auto Force = torch::mv(cloth.M, cloth.nodes_gravity);	// Gravity Force

	Tensor Jaco = add_internal_forces(cloth, ns_t, Force);

	Force = Force - ::magic.handle_stiffness * torch::mv(cloth.handle_indices_t, ns_t - cloth.init_pos_t);

	Jaco = Jaco - ::magic.handle_stiffness * cloth.handle_indices_t;

	Tensor dv = EigenSolver::apply(cloth.M - dt * dt * Jaco, dt * (Force + dt * torch::mv(Jaco, vs_t))).reshape({ num_nodes, 3 });

	for (int n = 0; n < num_nodes; ++n)
	{
		Node* node = cloth.mesh.nodes[n];

		node->v = node->v + dv[n];
		node->x = node->x + node->v * dt;
		node->acceleration = dv[n] / dt;
	}
}

void implicit_update_cuda(Cloth& cloth, const Tensor& dt_cpu, const Tensor& dt_cuda)
{	
	int num_nodes{ static_cast<int>(cloth.mesh.nodes.size()) };

	auto Force = cloth.gravity_cuda.clone();

	std::vector<Tensor> ns, vs;

	for (int n = 0; n < num_nodes; ++n)
	{
		const Node* node = cloth.mesh.nodes[n];

		ns.push_back(cloth.mesh.nodes[n]->x);

		vs.push_back(cloth.mesh.nodes[n]->v);
	}

	Tensor ns_t = torch::cat(ns).cuda();

	Tensor vs_t = torch::cat(vs).cuda();

	Tensor Jaco = add_internal_forces_cuda(cloth, ns_t, Force, TNOPT_CUDA);

	Force -= (cloth.handle_stiffness_cuda * torch::mv(cloth.handle_indices_t_cuda, ns_t - cloth.init_pos_t_cuda));

	Jaco -= (cloth.handle_stiffness_cuda * cloth.handle_indices_t_cuda);

	Tensor A = cloth.M_cuda - ((dt_cuda * dt_cuda) * Jaco);

	Tensor b = dt_cuda * (Force + dt_cuda * torch::mv(Jaco, vs_t));

	Tensor dv = EigenSolverCUDA::apply(A, b, 0).reshape({ num_nodes, 3 });

#pragma omp parallel for
	for (int n = 0; n < num_nodes; ++n)
	{
		Node* node = cloth.mesh.nodes[n];

		node->v = node->v + dv[n];
		node->x = node->x + node->v * dt_cpu;
	}
}

void implicit_update_cuda(Cloth& cloth, const Tensor& ns_t_cuda, const Tensor& vs_t_cuda, const Tensor& dt, const int& device_idx, const torch::TensorOptions& tensor_opt)
{
	int num_nodes{ static_cast<int>(cloth.mesh.nodes.size()) };

	auto Force = cloth.gravity_cuda.clone();

	Tensor Jaco = add_internal_forces_cuda(cloth, ns_t_cuda, Force, tensor_opt);

	Force -= cloth.handle_stiffness_cuda * torch::mv(cloth.handle_indices_t_cuda, ns_t_cuda - cloth.init_pos_t_cuda);

	Jaco -= cloth.handle_stiffness_cuda * cloth.handle_indices_t_cuda;

	Tensor A = cloth.M_cuda - ((dt * dt) * Jaco);

	Tensor b = dt * (Force + dt * torch::mv(Jaco, vs_t_cuda));

	Tensor dv_cuda = EigenSolverCUDA::apply(A, b, device_idx).to(torch::Device(torch::kCUDA, device_idx));

	cloth.vel_cuda = vs_t_cuda + dv_cuda;

	cloth.pos_cuda = ns_t_cuda + cloth.vel_cuda * dt;
}
