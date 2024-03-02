#include <random>
#include <algorithm>
#include <omp.h>

#include "Proximity.h"
#include "Geometry.h"

struct Min_Face
{
	Tensor key;
	Face* val;

	Min_Face(): key(INFINITY_TENSOR), val() {}
	
	void add(Tensor key, Face* val)
	{
		if ((key < this->key).item<bool>())
		{
			this->key = key;
			this->val = val;
		}
	}
};

struct Min_Edge
{
	Tensor key;
	Edge* val;

	Min_Edge() : key(INFINITY_TENSOR), val() {}

	void add(Tensor key, Edge* val)
	{
		if ((key < this->key).item<bool>())
		{
			this->key = key;
			this->val = val;
		}
	}
};

struct Min_Node
{
	Tensor key;
	Node* val;

	Min_Node() : key(INFINITY_TENSOR), val() {}

	void add(Tensor key, Node* val)
	{
		if ((key < this->key).item<bool>())
		{
			this->key = key;
			this->val = val;
		}
	}
};

inline int NEXT(int i)
{
	return (i < 2) ? (i + 1) : (i - 2);
}

inline int PREV(int i)
{
	return (i > 0) ? (i - 1) : (i + 2);
}

static std::vector<Min_Face> node_prox[2];
static std::vector<Min_Edge> edge_prox[2];
static std::vector<Min_Node> face_prox[2];

std::vector<Constraint*> Proximity::proximity_constraints(const std::vector<Mesh*>& _meshes,
	const std::vector<Mesh*>& _obs_meshes,
	Tensor mu, Tensor mu_obs)
{
	//std::cout << "Get Proximity Constraints" << std::endl;
	
	std::vector<Constraint*> cons;

	int nthreads = omp_get_max_threads();

	if (!prox_faces)
		prox_faces = new std::vector<std::pair<Face const*, Face const*>>[nthreads];

	meshes = _meshes;
	obs_meshes = _obs_meshes;

	const Tensor dmin = 2 * ::magic.repulsion_thickness;

	std::vector<AccelStruct*> accs = create_accel_structs(meshes, false);
	std::vector<AccelStruct*> obs_accs = create_accel_structs(obs_meshes, false);

	int num_nodes = 0;
	int num_edges = 0;
	int num_faces = 0;

	for (int m = 0; m < meshes.size(); ++m)
	{
		num_nodes = num_nodes + static_cast<int>(meshes[m]->nodes.size());
		num_edges = num_edges + static_cast<int>(meshes[m]->edges.size());
		num_faces = num_faces + static_cast<int>(meshes[m]->faces.size());
	}

	for (int i = 0; i < 2; ++i)
	{
		::node_prox[i].assign(num_nodes, Min_Face());
		::edge_prox[i].assign(num_edges, Min_Edge());
		::face_prox[i].assign(num_faces, Min_Node());
	}

	for_overlapping_faces(accs, obs_accs, dmin, true);

	std::vector<std::pair<Face const*, Face const*> > tot_faces;
	for (int t = 0; t < nthreads; ++t)
		tot_faces.insert(tot_faces.end(), prox_faces[t].begin(), prox_faces[t].end());

	// random_shuffle deprecated
	std::random_device rng;
	std::mt19937 urng(rng());
	std::shuffle(tot_faces.begin(), tot_faces.end(), urng);

	for (int i = 0; i < tot_faces.size(); ++i)
		compute_proximities(tot_faces[i].first, tot_faces[i].second);

	for (int n = 0; n < num_nodes; n++)
		for (int i = 0; i < 2; i++) 
		{
			Min_Face& m = ::node_prox[i][n];
			if ((m.key < dmin).item<bool>())
				cons.push_back(make_constraint(get_node(n, meshes), m.val, mu, mu_obs));
		}

	for (int e = 0; e < num_edges; e++)
		for (int i = 0; i < 2; i++) 
		{
			Min_Edge& m = ::edge_prox[i][e];
			if ((m.key < dmin).item<bool>())
				cons.push_back(make_constraint(get_edge(e, meshes), m.val, mu, mu_obs));
		}

	for (int f = 0; f < num_faces; f++)
		for (int i = 0; i < 2; i++) 
		{
			Min_Node& m = ::face_prox[i][f];
			if ((m.key < dmin).item<bool>())
				cons.push_back(make_constraint(m.val, get_face(f, meshes), mu, mu_obs));
		}

	destroy_accel_structs(accs);
	destroy_accel_structs(obs_accs);

	//if (cons.size() > 0)
	//{
	//	std::cout << "Constraints size: " << cons.size() << std::endl;
	//}

	return cons;
}

void Proximity::compute_proximities(const Face* face0, const Face* face1)
{
	kDOP18 nb[6], eb[6], fb[2];

	for (int v = 0; v < 3; ++v) {
		nb[v] = node_box(face0->v[v]->node, false);
		nb[v + 3] = node_box(face1->v[v]->node, false);
	}

	for (int v = 0; v < 3; ++v) {
		eb[v] = nb[NEXT(v)] + nb[PREV(v)];//edge_box(face0->adje[v], true);//
		eb[v + 3] = nb[NEXT(v) + 3] + nb[PREV(v) + 3];//edge_box(face1->adje[v], true);//
	}

	fb[0] = nb[0] + nb[1] + nb[2];
	fb[1] = nb[3] + nb[4] + nb[5];
	double thick = 2 * ::magic.repulsion_thickness.item<double>();

	for (int v = 0; v < 3; v++) {
		if (!overlap(nb[v], fb[1], thick))
			continue;
		add_proximity(face0->v[v]->node, face1, thick);
	}
	for (int v = 0; v < 3; v++) {
		if (!overlap(nb[v + 3], fb[0], thick))
			continue;
		add_proximity(face1->v[v]->node, face0, thick);
	}
	for (int e0 = 0; e0 < 3; e0++)
		for (int e1 = 0; e1 < 3; e1++) {
			if (!overlap(eb[e0], eb[e1 + 3], thick))
				continue;
			add_proximity(face0->adj_edges[e0], face1->adj_edges[e1], thick);
		}
}

bool Proximity::is_movable(const Node* n)
{
	return find_node_in_meshes(n, meshes) != -1;
}

bool Proximity::is_movable(const Edge* e)
{
	return find_edge_in_meshes(e, meshes) != -1;
}

bool Proximity::is_movable(const Face* f)
{
	return find_face_in_meshes(f, meshes) != -1;
}

Node* Proximity::get_node(int i, const std::vector<Mesh*>& meshes)
{
	for (int m = 0; m < meshes.size(); ++m)
	{
		const std::vector<Node*>& nodes = meshes[m]->nodes;
		if (i < nodes.size())
			return nodes[i];
		else
			i -= nodes.size();
	}
	return nullptr;
}

Edge* Proximity::get_edge(int i, const std::vector<Mesh*>& meshes)
{
	for (int m = 0; m < meshes.size(); ++m)
	{
		const std::vector<Edge*>& edges = meshes[m]->edges;
		if (i < edges.size())
			return edges[i];
		else
			i -= edges.size();
	}
	return nullptr;
}

Face* Proximity::get_face(int i, const std::vector<Mesh*>& meshes)
{
	for (int m = 0; m < meshes.size(); ++m)
	{
		const std::vector<Face*>& faces = meshes[m]->faces;
		if (i < faces.size())
			return faces[i];
		else
			i -= faces.size();
	}
	return nullptr;
}


Tensor Proximity::area(const Node* node)
{
	if (is_movable(node))
		return node->a;

	Tensor a = ZERO;
	for (int v = 0; v < node->verts.size(); v++)
		for (int f = 0; f < node->verts[v]->adj_faces.size(); f++)
			a = a + area(node->verts[v]->adj_faces[f]) / 3;

	return a;
}

Tensor Proximity::area(const Edge* edge)
{
	Tensor a = ZERO;

	if (edge->adj_faces[0])
		a = a + area(edge->adj_faces[0]) / 3;
	if (edge->adj_faces[1])
		a = a + area(edge->adj_faces[1]) / 3;

	return a;
}

Tensor Proximity::area(const Face* face)
{
	if (is_movable(face))
		return face->w_a;

	const Tensor& x0 = face->v[0]->node->x;
	const Tensor& x1 = face->v[1]->node->x;
	const Tensor& x2 = face->v[2]->node->x;

	return torch::norm(torch::cross(x1 - x0, x2 - x0)) / 2;
}

int Proximity::get_node_index(const Node* n, const std::vector<Mesh*>& meshes)
{
	int i = 0;
	for (int m = 0; m < meshes.size(); ++m)
	{
		const std::vector<Node*>& nodes = meshes[m]->nodes;
		if (n->index < nodes.size() && n == nodes[n->index])
			return i + n->index;
		else
			i += nodes.size();
	}

	return -1;
}

int Proximity::get_edge_index(const Edge* e, const std::vector<Mesh*>& meshes)
{
	int i = 0;
	for (int m = 0; m < meshes.size(); ++m)
	{
		const std::vector<Edge*>& edges = meshes[m]->edges;
		if (e->index < edges.size() && e == edges[e->index])
			return i + e->index;
		else
			i += edges.size();
	}
	return -1;
}

int Proximity::get_face_index(const Face* f, const std::vector<Mesh*>& meshes)
{
	int i = 0;
	for (int m = 0; m < meshes.size(); ++m)
	{
		const std::vector<Face*>& faces = meshes[m]->faces;
		if (f->index < faces.size() && f == faces[f->index])
			return i + f->index;
		else
			i += faces.size();
	}
	return -1;
}

void Proximity::add_proximity(const Node* node, const Face* face, double thick)
{
	if (node == face->v[0]->node ||
		node == face->v[1]->node ||
		node == face->v[2]->node)
		return;

	Tensor n;
	Tensor w[4];
	n = ZERO3;
	bool over = false;
	Tensor d = signed_vf_distance(node->x, face->v[0]->node->x,
		face->v[1]->node->x, face->v[2]->node->x,
		&n, w, thick, over);
	if (over) 
		return;

	d = abs(d);
	bool inside = (min(min(-w[1], -w[2]), -w[3]) >= -1e-6).item<bool>();
	if (!inside)
		return;

	if (is_movable(node)) {
		int side = (torch::dot(n, node->n) >= 0).item<bool>() ? 0 : 1;
		::node_prox[side][get_node_index(node, meshes)].add(d, (Face*)face);
	}
	if (is_movable(face)) {
		int side = (torch::dot(-n, face->n) >= 0).item<bool>() ? 0 : 1;
		::face_prox[side][get_face_index(face, meshes)].add(d, (Node*)node);
	}
}

bool Proximity::in_wedge(Tensor w, const Edge* edge0, const Edge* edge1)
{
	Tensor x = (1 - w) * edge0->nodes[0]->x + w * edge0->nodes[1]->x;
	bool in = true;
	for (int s = 0; s < 2; s++) {
		const Face* face = edge1->adj_faces[s];
		if (!face)
			continue;
		const Node* node0 = edge1->nodes[s], * node1 = edge1->nodes[1 - s];
		Tensor e = node1->x - node0->x, n = face->n, r = x - node0->x;
		in &= (stp(e, n, r) >= 0).item<bool>();
	}
	return in;
}

void Proximity::add_proximity(const Edge* edge0, const Edge* edge1, double thick)
{
	if (edge0->nodes[0] == edge1->nodes[0] || 
		edge0->nodes[0] == edge1->nodes[1] ||
		edge0->nodes[1] == edge1->nodes[0] || 
		edge0->nodes[1] == edge1->nodes[1])
		return;
	//if (!overlap(edge_box(edge0, true), edge_box(edge1, true), ::magic.repulsion_thickness.item<double>()))
	//    return;
	Tensor n;
	Tensor w[4];
	w[0] = w[1] = w[2] = w[3] = ZERO;
	n = ZERO3;
	bool over = false;
	Tensor d = signed_ee_distance(edge0->nodes[0]->x, edge0->nodes[1]->x,
		edge1->nodes[0]->x, edge1->nodes[1]->x, &n, w, thick, over);
	if (over) 
		return;
	d = abs(d);

	bool inside = ((min(min(w[0], w[1]), min(-w[2], -w[3])) >= -1e-6).item<int>()
		&& in_wedge(w[1], edge0, edge1)
		&& in_wedge(-w[3], edge1, edge0));
	if (!inside)
		return;
	//cout << "good" << endl;
	if (is_movable(edge0)) {
		Tensor edge0n = edge0->nodes[0]->n + edge0->nodes[1]->n;
		int side = (dot(n, edge0n) >= 0).item<int>() ? 0 : 1;
		::edge_prox[side][get_edge_index(edge0, meshes)].add(d, (Edge*)edge1);
	}
	if (is_movable(edge1)) {
		Tensor edge1n = edge1->nodes[0]->n + edge1->nodes[1]->n;
		int side = (dot(-n, edge1n) >= 0).item<int>() ? 0 : 1;
		::edge_prox[side][get_edge_index(edge1, meshes)].add(d, (Edge*)edge0);
	}
}

void Proximity::find_proximities(const Face* face0, const Face* face1) 
{
	int t = omp_get_thread_num();
	prox_faces[t].push_back(std::make_pair(face0, face1));
}

void Proximity::for_overlapping_faces(BVHNode* node, double thickness)
{
	if (node->isLeaf() || !node->_active)
		return;
	for_overlapping_faces(node->getLeftChild(), thickness);
	for_overlapping_faces(node->getRightChild(), thickness);
	for_overlapping_faces(node->getLeftChild(), node->getRightChild(), thickness);
}

void Proximity::for_overlapping_faces(BVHNode* node0, BVHNode* node1, double thickness)
{
	if (!node0->_active && !node1->_active)
		return;
	if (!overlap(node0->_box, node1->_box, thickness))
		return;
	if (node0->isLeaf() && node1->isLeaf()) {
		Face* face0 = node0->getFace(),
			* face1 = node1->getFace();
		find_proximities(face0, face1);
	}
	else if (node0->isLeaf()) {
		for_overlapping_faces(node0, node1->getLeftChild(), thickness);
		for_overlapping_faces(node0, node1->getRightChild(), thickness);
	}
	else {
		for_overlapping_faces(node0->getLeftChild(), node1, thickness);
		for_overlapping_faces(node0->getRightChild(), node1, thickness);
	}
}

void Proximity::for_overlapping_faces(const std::vector<AccelStruct*>& accs, const std::vector<AccelStruct*>& obs_accs, Tensor thickness0, bool parallel)
{
	omp_set_num_threads(1);
	double thickness = thickness0.item<double>();
	int nnodes = (int)ceil(sqrt(2 * omp_get_max_threads()));
	std::vector<BVHNode*> nodes = collect_upper_nodes(accs, nnodes);
	int nthreads = omp_get_max_threads();
	omp_set_num_threads(parallel ? omp_get_max_threads() : 1);
#pragma omp parallel for
	for (int n = 0; n < nodes.size(); n++) {
		for_overlapping_faces(nodes[n], thickness);
		for (int m = 0; m < n; m++)
			for_overlapping_faces(nodes[n], nodes[m], thickness);
		for (int o = 0; o < obs_accs.size(); o++)
			if (obs_accs[o]->root)
				for_overlapping_faces(nodes[n], obs_accs[o]->root, thickness);
	}

	nodes = collect_upper_nodes(obs_accs, nnodes);
#pragma omp parallel for
	for (int n = 0; n < nodes.size(); n++) {
		for_overlapping_faces(nodes[n], thickness);
		for (int m = 0; m < n; m++)
			for_overlapping_faces(nodes[n], nodes[m], thickness);
	}

	omp_set_num_threads(nthreads);
}

std::vector<BVHNode*> Proximity::collect_upper_nodes(const std::vector<AccelStruct*>& accs, int nnodes) {

	std::vector<BVHNode*> nodes;
	for (int a = 0; a < accs.size(); a++)
		if (accs[a]->root)
			nodes.push_back(accs[a]->root);
	while (nodes.size() < nnodes) {
		std::vector<BVHNode*> children;
		for (int n = 0; n < nodes.size(); n++)
			if (nodes[n]->isLeaf())
				children.push_back(nodes[n]);
			else {
				children.push_back(nodes[n]->_left);
				children.push_back(nodes[n]->_right);
			}
		if (children.size() == nodes.size())
			break;
		nodes = children;
	}
	return nodes;
}

Constraint* Proximity::make_constraint(const Node* node, const Face* face, Tensor mu, Tensor mu_obs)
{
	IneqCon* con = new IneqCon;
	con->nodes[0] = (Node*)node;
	con->nodes[1] = (Node*)face->v[0]->node;
	con->nodes[2] = (Node*)face->v[1]->node;
	con->nodes[3] = (Node*)face->v[2]->node;
	for (int n = 0; n < 4; n++)
		con->free[n] = is_movable(con->nodes[n]);
	Tensor a = min(area(node), area(face));
	con->stiff = ::magic.collision_stiffness * a;
	bool over;
	Tensor d = signed_vf_distance(con->nodes[0]->x, con->nodes[1]->x,
		con->nodes[2]->x, con->nodes[3]->x,
		&con->n, con->w, 100, over);

	if ((d < 0).item<bool>())
		con->n = -con->n;

	con->mu = (!is_movable(node) || !is_movable(face)) ? mu_obs : mu;
	//do_detach(con);
	return con;
}

Constraint* Proximity::make_constraint(const Edge* edge0, const Edge* edge1, Tensor mu, Tensor mu_obs)
{
	IneqCon* con = new IneqCon;
	con->nodes[0] = (Node*)edge0->nodes[0];
	con->nodes[1] = (Node*)edge0->nodes[1];
	con->nodes[2] = (Node*)edge1->nodes[0];
	con->nodes[3] = (Node*)edge1->nodes[1];
	for (int n = 0; n < 4; n++)
		con->free[n] = is_movable(con->nodes[n]);
	Tensor a = min(area(edge0), area(edge1));
	con->stiff = ::magic.collision_stiffness * a;
	bool over;
	Tensor d = signed_ee_distance(con->nodes[0]->x, con->nodes[1]->x,
		con->nodes[2]->x, con->nodes[3]->x,
		&con->n, con->w, 100, over);

	if ((d < 0).item<bool>())
		con->n = -con->n;

	con->mu = (!is_movable(edge0) || !is_movable(edge1)) ? mu_obs : mu;
	//do_detach(con);
	return con;
}
