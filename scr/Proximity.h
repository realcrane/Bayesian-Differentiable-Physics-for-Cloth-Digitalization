#pragma once
#include "Constraint.h"
#include "CollisionUtil.h"

struct Proximity
{
	std::vector<std::pair<Face const*, Face const*>>* prox_faces = nullptr;

	std::vector<Mesh*> meshes, obs_meshes;

	std::vector<Constraint*> proximity_constraints(const std::vector<Mesh*>& _meshes,
		const std::vector<Mesh*>& _obs_meshes,
		Tensor mu, Tensor mu_obs);

	bool is_movable(const Node* n);

	bool is_movable(const Edge* e);

	bool is_movable(const Face* f);

	bool in_wedge(Tensor w, const Edge* edge0, const Edge* edge1);

	Tensor stp(Tensor u, Tensor v, Tensor w)
	{
		return torch::dot(u, torch::cross(v, w));
	}

	Tensor area(const Node* node);

	Tensor area(const Edge* edge);

	Tensor area(const Face* face);

	Node* get_node(int i, const std::vector<Mesh*>& meshes);

	Edge* get_edge(int i, const std::vector<Mesh*>& meshes);

	Face* get_face(int i, const std::vector<Mesh*>& meshes);

	int get_node_index(const Node* n, const std::vector<Mesh*>& meshes);

	int get_edge_index(const Edge* e, const std::vector<Mesh*>& meshes);

	int get_face_index(const Face* f, const std::vector<Mesh*>& meshes);

	void find_proximities(const Face* face0, const Face* face1);

	void compute_proximities(const Face* face0, const Face* face1);

	void add_proximity(const Node* node, const Face* face, double thick);

	void add_proximity(const Edge* edge0, const Edge* edge1, double thick);

	std::vector<BVHNode*> collect_upper_nodes(const std::vector<AccelStruct*>& accs, int n);

	void for_overlapping_faces(BVHNode* node, double thickness);

	void for_overlapping_faces(BVHNode* node0, BVHNode* node1, double thickness);

	void for_overlapping_faces(const std::vector<AccelStruct*>& accs, const std::vector<AccelStruct*>& obs_accs, Tensor thickness, bool parallel = true);

	Constraint* make_constraint(const Node* node, const Face* face, Tensor mu, Tensor mu_obs);

	Constraint* make_constraint(const Edge* edge0, const Edge* edge1, Tensor mu, Tensor mu_obs);
};