#pragma once

#include <vector>
#include <torch/torch.h>

#include "Constants.h"

using namespace torch::indexing;

using torch::Tensor;

struct Vert;
struct Face;

struct Node;
struct Edge;


struct Vert
{
	int label;
	
	Tensor u; // material space

	Tensor a; // area

	Tensor m; // mass

	Node* node;

	int index;	// position in mesh.verts

	std::vector<Face*> adj_faces;	// adjacent faces

	Vert(const Tensor& x, int label = 0) :
		u(x.index({ Slice(0, 2) })), label(label) 
	{
		a = ZERO;
	}
};

struct Node
{
	int label;
	
	Tensor x, x0;	// Node's current position and previous position

	Tensor v;	// Node's velocity

	Tensor m;	// Node's mass

	Tensor acceleration;

	Tensor a;	// Node's area

	Tensor n;	// Node's normal

	std::vector<Vert*> verts;	// A node can be shared by multiple vertices

	std::vector<Edge*> adj_egdes;	// adjacent edges

	int index; // position in mesh.nodes

	Node() = default;

	Node(const Tensor& x, const Tensor& v, int label = 0) :
		label(label), x(x), x0(x), v(v) 
	{
		a = ZERO;

		n = ZERO3;
	}
};

struct Edge
{
	int label;
	
	std::array<Node*, 2> nodes;	// two end nodes

	std::array<Face*, 2> adj_faces;	// adjacent Faces

	int index;	// position in mesh.edges

	Tensor theta;	// actual dihedral angle

	Tensor reference_angle;	// just to get sign of dihedral_angle() right

	Tensor theta_ideal, damage;	// rest dihedral angle, damage parameter

	Tensor l, ldaa, bias_angle;	// Terms for compute bending Force

	Tensor s0, eps_dot_0, fric_sig_0;	// Bending Friction Stress in previous step

	Edge() = default;

	Edge(Node* node0, Node* node1, int label = 0)
	{
		nodes[0] = node0;
		nodes[1] = node1;

		label = label;

		// member "theta" is computed in the compute_ms_data(Edge* edge) function.

		reference_angle = ZERO;
		theta_ideal = ZERO;
		damage = ZERO;

		s0 = ZERO;
		eps_dot_0 = ZERO;
		fric_sig_0 = ZERO;
	}
		
};

struct Face
{
	int label;

	int index; // position in mesh.faces
	
	std::array<Vert*, 3> v; // Vertices

	std::array<Edge*, 3> adj_edges;	// adjacent edges
	
	Tensor m; // Face Mass
	Tensor m_a; // Face Area in Material Space
	Tensor w_a; // Face Area in World Space

	Tensor s0, eps_dot_0, fric_sig_0; // Stretch friction Stress In previous step

	Tensor Dm, invDm;

	Tensor n; // Face normal

	Face() = default;

	Face(Vert* v1, Vert* v2, Vert* v3, int _label = 0)
	{
		v[0] = v1;
		v[1] = v2;
		v[2] = v3;

		label = _label;

		m = ZERO;
		m_a = ZERO;
		w_a = ZERO;

		s0 = ZERO3;
		eps_dot_0 = ZERO3;
		fric_sig_0 = ZERO3;
	}
};

struct Mesh
{
	bool is_cloth;

	std::vector<Vert*> verts;
	std::vector<Node*> nodes;
	std::vector<Face*> faces;
	std::vector<Edge*> edges;

	void add(Vert* vert);
	void add(Node* node);
	void add(Edge* edge);
	void add(Face* face);

	Edge* get_edge(const Node* n0, const Node* n1);
	void add_edges_if_needed(const Face* face);

	void remove(Vert* vert);
	void remove(Node* node);
	void remove(Edge* edge);
	void remove(Face* face);

	void update_x0();

	void compute_ms_data();
	void compute_ms_data(Face* face);
	void compute_ms_data(Edge* edge);
	void compute_ms_data(Vert* vert);
	void compute_ms_data(Node* node);

	void compute_ws_data();
	void compute_ws_data(Face* face);
	void compute_ws_data(Edge* edge);
	void compute_ws_data(Node* node);

};

int num_bending_edge(const Mesh& mesh);	// Count edges where bending force exist

void delete_mesh(Mesh& mesh);

void clear_mesh(Mesh& mesh);

Vert* edge_vert(const Edge* edge, int side, int i);

Vert* edge_opp_vert(const Edge* edge, int side);

