#include "Mesh.h"
#include "Geometry.h"

using torch::Tensor;

inline int tri_idx_next(int i)
{
	return i == 2 ? 0 : i + 1;
}

inline int tri_idx_prev(int i)
{
	return i == 0 ? 2 : i - 1;
}

void Mesh::update_x0()
{
	// Update Nodes' previous position
	
	for (auto& n : nodes)
		n->x0 = n->x;
}

void Mesh::add(Vert* vert) {

	verts.push_back(vert);

	vert->node = nullptr;

	vert->index = verts.size() - 1;

	vert->adj_faces.clear();
	
}

void Mesh::add(Node* node){

	nodes.push_back(node);

	for (int v = 0; v < node->verts.size(); ++v)
		node->verts[v]->node = node;

	node->index = nodes.size() - 1;

	node->adj_egdes.clear();
}

void Mesh::add(Edge* edge){

	edges.push_back(edge);

	edge->adj_faces[0] = nullptr;

	edge->adj_faces[1] = nullptr;

	edge->index = edges.size() - 1;

	auto find_edge_0 = find(edge->nodes[0]->adj_egdes.cbegin(), edge->nodes[0]->adj_egdes.cend(), edge);

	if (find_edge_0 == edge->nodes[0]->adj_egdes.cend())
		edge->nodes[0]->adj_egdes.push_back(edge);

	auto find_edge_1 = find(edge->nodes[1]->adj_egdes.cbegin(), edge->nodes[1]->adj_egdes.cend(), edge);

	if (find_edge_1 == edge->nodes[1]->adj_egdes.cend())
		edge->nodes[1]->adj_egdes.push_back(edge);
}

void Mesh::add(Face* face){

	faces.push_back(face);

	face->index = faces.size() - 1;

	add_edges_if_needed(face);

	for (int i = 0; i < 3; ++i)
	{
		Vert* v0 = face->v[tri_idx_next(i)];
		Vert* v1 = face->v[tri_idx_prev(i)];

		auto find_face = find(v0->adj_faces.cbegin(), v0->adj_faces.cend(), face);

		if (find_face == v0->adj_faces.cend())
			v0->adj_faces.push_back(face);

		Edge* e = get_edge(v0->node, v1->node);
		
		face->adj_edges[i] = e;
		
		int size = e->nodes[0] == v0->node ? 0 : 1;	// The order of adjacent faces are important
		
		e->adj_faces[size] = face;
	}
}

Edge* Mesh::get_edge(const Node* n0, const Node* n1)
{
	for (Edge* e : n0->adj_egdes)
	{
		if (e->nodes[0] == n1 || e->nodes[1] == n1)
			return e;
	}

	return nullptr;
}

void Mesh::add_edges_if_needed(const Face* face)
{
	for (int i = 0; i < 3; ++i)
	{
		Node* n0 = face->v[i]->node;
		Node* n1 = face->v[tri_idx_next(i)]->node;

		if (get_edge(n0, n1) == nullptr)
			this->add(new Edge(n0, n1));
	}
}

void Mesh::compute_ms_data()
{
	for (auto f : faces)
		compute_ms_data(f);
	for (auto e : edges)
		compute_ms_data(e);
	for (auto v : verts)
		compute_ms_data(v);
	for (auto n : nodes)
		compute_ms_data(n);
}

void Mesh::compute_ws_data()
{
	for (auto f : faces)
		compute_ws_data(f);
	for (auto e : edges)
		compute_ws_data(e);
	for (auto n : nodes)
		compute_ws_data(n);
}

void Mesh::compute_ms_data(Face* face)
{
	const Tensor& v0 = face->v[0]->u;
	const Tensor& v1 = face->v[1]->u;
	const Tensor& v2 = face->v[2]->u;

	face->Dm = torch::stack({ v1 - v0, v2 - v0 }, 1);

	face->m_a = 0.5 * torch::det(face->Dm);

	if (face->m_a.item<double>() == 0.0)
		face->invDm = torch::zeros({ 2,2 }, TNOPT);
	else
		face->invDm = face->Dm.inverse();

}

void Mesh::compute_ms_data(Edge* edge)
{
	edge->l = ZERO;

	for(int s = 0; s < 2; ++s)
		if (edge->adj_faces[s] != nullptr)
			edge->l = edge->l + torch::norm(edge_vert(edge, s, 0)->u - edge_vert(edge, s, 1)->u);
	
	if (edge->adj_faces[0] && edge->adj_faces[1])
		edge->l = edge->l / 2;

	if (!edge->adj_faces[0] || !edge->adj_faces[1])
		return;

	edge->ldaa = edge->l / (edge->adj_faces[0]->m_a + edge->adj_faces[1]->m_a);
	
	Tensor du0 = edge_vert(edge, 0, 1)->u - edge_vert(edge, 0, 0)->u;
	edge->bias_angle = torch::atan2(du0[1], du0[0]) * (4 / M_PI) - 1;
}

void Mesh::compute_ms_data(Vert* vert)
{
	// Vert area equals to sum of the 1/3 of face's area s
	for (const Face* f : vert->adj_faces)
		vert->a = vert->a + (f->m_a / 3.0);
}

void Mesh::compute_ms_data(Node* node)
{
	// Node area equal the sum of vertices' area
	for (const Vert* v : node->verts)
		node->a = node->a + v->a;
}

void Mesh::compute_ws_data(Face* face)
{
	const Tensor& x0 = face->v[0]->node->x;
	const Tensor& x1 = face->v[1]->node->x;
	const Tensor& x2 = face->v[2]->node->x;

	Tensor cross_norm = torch::norm(torch::cross(x1 - x0, x2 - x0));

	face->w_a = 0.5 * cross_norm;
	face->n = torch::cross(x1 - x0, x2 - x0) / cross_norm;
}

void Mesh::compute_ws_data(Edge* edge)
{
	edge->theta = dihedral_angle(edge);
}

void Mesh::compute_ws_data(Node* node)
{
	for (const Vert* v : node->verts)
		for (const Face* f : v->adj_faces)
			node->n = node->n + f->n;

	if (torch::norm(node->n).item<float>() != 0.0)
		node->n = node->n / torch::norm(node->n);
}

Vert* edge_vert(const Edge* edge, int side, int i)
{
	Face* face = edge->adj_faces[side];

	if (face == nullptr)
		return nullptr;

	for (int v = 0; v < 3; ++v)
		if (face->v[v]->node == edge->nodes[i])
			return face->v[v];

	return nullptr;
}

Vert* edge_opp_vert(const Edge* edge, int side)
{
	Face* face = edge->adj_faces[side];

	if (face == nullptr)
		return nullptr;
	
	for (int v = 0; v < 3; ++v)
		if (face->v[v]->node == edge->nodes[side])
			return face->v[tri_idx_prev(v)];

	return nullptr;
}

int num_bending_edge(const Mesh& mesh)
{
	int num_edges{ 0 };

	for (const auto& e : mesh.edges) {
		if (e->adj_faces[0] == nullptr || e->adj_faces[1] == nullptr) {
			continue;
		}
		else {
			++num_edges;
		}
	}

	return num_edges;
}

void delete_mesh(Mesh& mesh)
{
	for (int v = 0; v < mesh.verts.size(); v++)
		delete mesh.verts[v];
	for (int n = 0; n < mesh.nodes.size(); n++)
		delete mesh.nodes[n];
	for (int e = 0; e < mesh.edges.size(); e++)
		delete mesh.edges[e];
	for (int f = 0; f < mesh.faces.size(); f++)
		delete mesh.faces[f];

	mesh.verts.clear();
	mesh.nodes.clear();
	mesh.edges.clear();
	mesh.faces.clear();
}

void clear_mesh(Mesh& mesh)
{
	mesh.verts.clear();
	mesh.nodes.clear();
	mesh.edges.clear();
	mesh.faces.clear();
}