#include "Cloth.h"
#include "Constants.h"
#include "DDE.h"
#include "MeshIO.h"	

void Cloth::compute_masses()
{
	for (int v = 0; v < mesh.verts.size(); ++v)
		mesh.verts[v]->m = ZERO;

	for (int n = 0; n < mesh.nodes.size(); ++n)
		mesh.nodes[n]->m = ZERO;

	for (int f = 0; f < mesh.faces.size(); ++f)
	{
		Face* face = mesh.faces[f];

		face->m = face->m_a * density[face->label];

		for (int v = 0; v < 3; ++v)
		{
			face->v[v]->m = face->v[v]->m + face->m / 3.0;
			face->v[v]->node->m = face->v[v]->node->m + face->m / 3.0;
		}
	}

	std::vector<Tensor> ms;

	for (int n = 0; n < mesh.nodes.size(); ++n)
	{
		const Node* node = mesh.nodes[n];
		ms.push_back(mesh.nodes[n]->m.squeeze());
	}

	M = torch::diag(torch::stack(ms).repeat_interleave(3));
}

void Cloth::compute_masses_cuda()
{		
	M_cuda = torch::diag(torch::mv(node_m_matrix, torch::cat(density) * faces_m_cuda).repeat_interleave(3));
	
	gravity_cuda = torch::mv(M_cuda, nodes_gravity);
}

void Cloth::set_parameters(const Tensor& _density, const Tensor& _stretching, const Tensor& _bending, const Tensor& _damping)
{
	density.clear();
	
	density.push_back(_density);

	stretch_ori = _stretching;

	bending = _bending;

	damping = _damping;
}

void Cloth::set_parameters(const std::vector<Tensor>& _density, const std::vector<Tensor>& _stretching, const std::vector<Tensor>& _bending, const std::vector<Tensor>& _damping)
{	
	// Alter face label for heterogeneous cloth simulation
	for (int f = 0; f < mesh.faces.size(); ++f) {
		mesh.faces[f]->label = f;	
	}

	density.clear();

	density = _density; // Set density

	stretch_ori = torch::cat(_stretching);	// Set original stretching stiffness (unsampled)

	bending = torch::cat(_bending);	// Set bending

	damping = torch::stack(_damping);	// Set damping
}

void Cloth::set_parameters_cuda(const Tensor& _density, const Tensor& _stretching, const Tensor& _bending, const Tensor& _damping)
{
	density.clear();
	
	density.push_back(_density);

	stretch_ori_cuda = _stretching;

	bending_cuda = _bending;
}

void Cloth::set_parameters_cuda(const std::vector<Tensor>& _density, const std::vector<Tensor>& _stretching, const std::vector<Tensor>& _bending, const std::vector<Tensor>& _damping)
{
	int num_faces{ static_cast<int>(mesh.faces.size()) };

	// Alter face label for heterogeneous cloth simulation
	for (int f = 0; f < mesh.faces.size(); ++f) {
		mesh.faces[f]->label = f;
	}

	density.clear();	// Clear the density vector before setting (for checkpoint)

	density = _density; 	// Set density

	stretch_ori_cuda = torch::cat(_stretching);	// Set original stretching stiffness (unsampled) (CPU for saving memory)

	bending_cuda = torch::cat(_bending);	// Set bending

	std::cout << "Stretching ori shape (CUDA): " << stretch_ori_cuda.sizes() << std::endl;

	std::cout << "Bending shape: " << bending_cuda.sizes() << std::endl;
}

void Cloth::set_densities_cuda(const std::vector<Tensor>& _density)
{
	int num_faces{ static_cast<int>(mesh.faces.size()) };

	// Alter face label for heterogeneous cloth simulation
	for (int f = 0; f < mesh.faces.size(); ++f) {
		mesh.faces[f]->label = f;
	}

	density.clear();	// Clear the density vector before setting (for checkpoint)

	density = _density; 	// Set density
}

void Cloth::set_stretches_cuda(const std::vector<Tensor>& _stretching)
{
	stretch_ori_cuda = torch::cat(_stretching);	// Set original stretching stiffness (unsampled)	

	std::cout << "Stretching ori shape (CUDA): " << stretch_ori_cuda.sizes() << std::endl;
}

void Cloth::set_stretches_cuda(const std::vector<Tensor>& c11s, const std::vector<Tensor>& c12s, const std::vector<Tensor>& c22s, const std::vector<Tensor>& c33s)
{
	c11_cuda.clear();
	c12_cuda.clear();
	c22_cuda.clear();
	c33_cuda.clear();
	
	c11_cuda = c11s;
	c12_cuda = c12s;
	c22_cuda = c22s;
	c33_cuda = c33s;
}


void Cloth::set_bendings_cuda(const std::vector<Tensor>& _bending)
{
	bending_cuda = torch::cat(_bending);	// Set bending
}

std::pair<Tensor, Tensor> Cloth::get_pos_vel()
{
	std::vector<Tensor> ns, vs;

	for (int n = 0; n < mesh.nodes.size(); ++n)
	{
		const Node* node = mesh.nodes[n];

		ns.push_back(node->x);

		vs.push_back(node->v);
	}

	return std::make_pair(torch::cat(ns), torch::cat(vs));
}

std::pair<Tensor, Tensor> Cloth::faces_uv_idx()
{
	std::vector<int64_t> faces_indices;
	std::vector<Tensor> faces_uvs;

	for (const Face* f : mesh.faces) {
		for (int v = 0; v < 3; ++v) {
			faces_indices.push_back(static_cast<int64_t>(f->v[v]->index));
			faces_uvs.push_back(f->v[v]->u);
		}
	}
	
	return std::make_pair(torch::stack(faces_uvs).view({ -1, 3, 2 }), torch::tensor(faces_indices, torch::kInt64).view({-1, 3}));
}

std::pair<Tensor, Tensor> Cloth::get_pos_vel_cuda()
{
	return std::make_pair(pos_cuda, vel_cuda);
}

void Cloth::update_pos_vel_cuda(const int& device_idx)
{
	// Update the position and velocity vector from current cloth mesh (pos_cuda, vel_cuda)
	std::vector<Tensor> ns, vs;

	for (int n = 0; n < mesh.nodes.size(); ++n)
	{
		const Node* node = mesh.nodes[n];

		ns.push_back(node->x);

		vs.push_back(node->v);
	}

	pos_cuda = torch::cat(ns).to(torch::Device(torch::kCUDA, device_idx));
	vel_cuda = torch::cat(vs).to(torch::Device(torch::kCUDA, device_idx));
}

void Cloth::update_mesh_cuda(const Tensor& pos, const Tensor& vel)
{	
	// Update mesh nodes position and velocity from tensor pos and vel (Used for cuda version code)
	for (int n = 0; n < mesh.nodes.size(); ++n)	{
		mesh.nodes[n]->v = vel[n];
		mesh.nodes[n]->x = pos[n];
	}
}

void Cloth::update_mesh(const Tensor& pos, const Tensor& vel)
{
	/* The function for using checkpoint in Python 
	The pos and vel should be reshaped to [num_nodes, 3] in python */

	compute_masses();	// Compute node/face mass

	// Update mesh node position/velocity
	for (int n = 0; n < mesh.nodes.size(); ++n)	{
		mesh.nodes[n]->v = vel[n];
		mesh.nodes[n]->x = pos[n];
	}
}