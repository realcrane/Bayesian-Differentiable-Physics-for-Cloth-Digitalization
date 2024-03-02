#pragma once

#include "BVH.h"

using BVHNode = DeformBVHNode;
using BVHTree = DeformBVHTree;

struct AccelStruct
{
	BVHTree tree;
	BVHNode* root;

	std::vector<BVHNode*> leaves;

	AccelStruct(const Mesh& mesh, bool ccd);
};

void mark_all_inactive(AccelStruct& acc);

void mark_active(AccelStruct& acc, const Face* face);

void update_accel_struct(AccelStruct& acc);

std::vector<AccelStruct*> create_accel_structs (const std::vector<Mesh*>& meshes, bool ccd);

int find_node_in_meshes(const Node* n, const std::vector<Mesh*>& meshes);

int find_edge_in_meshes(const Edge* e, const std::vector<Mesh*>& meshes);

int find_face_in_meshes(const Face* f, const std::vector<Mesh*>& meshes);

void destroy_accel_structs(std::vector<AccelStruct*>& accs);
