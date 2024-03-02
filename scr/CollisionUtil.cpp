#include "CollisionUtil.h"

void collect_leaves(BVHNode* node, std::vector<BVHNode*>& leaves);

AccelStruct::AccelStruct(const Mesh& mesh, bool ccd):
    tree(const_cast<Mesh&>(mesh), ccd)
{
	root = tree._root;

	leaves = std::vector<BVHNode*>(mesh.faces.size());

	if (root != nullptr)
	{
        collect_leaves(root, leaves);
	}
}

void collect_leaves(BVHNode* node, std::vector<BVHNode*>& leaves) 
{
    if (node->isLeaf())
    {
        int f = node->getFace()->index;
        if (f >= leaves.size())
            leaves.resize(f + 1);
        leaves[f] = node;
    }
    else 
    {
        collect_leaves(node->getLeftChild(), leaves);
        collect_leaves(node->getRightChild(), leaves);
    }
}

std::vector<AccelStruct*> create_accel_structs(const std::vector<Mesh*>& meshes, bool ccd) 
{
    std::vector<AccelStruct*> accs(meshes.size());

    for (int m = 0; m < meshes.size(); m++)
    {
        accs[m] = new AccelStruct(*meshes[m], ccd);
    }

    return accs;
}

void mark_descendants(BVHNode* node, bool active);
void mark_ancestors(BVHNode* node, bool active);

void mark_all_inactive(AccelStruct& acc) 
{
    if (acc.root)
    {
        mark_descendants(acc.root, false);
    }
}

void mark_descendants(BVHNode* node, bool active) 
{
    node->_active = active;

    if (!node->isLeaf()) 
    {
        mark_descendants(node->_left, active);
        mark_descendants(node->_right, active);
    }
}

void mark_active(AccelStruct& acc, const Face* face) 
{
    if (acc.root)
    {
        mark_ancestors(acc.leaves[face->index], true);
    }
}

void mark_ancestors(BVHNode* node, bool active) 
{
    node->_active = active;
    if (!node->isRoot())
        mark_ancestors(node->_parent, active);
}

int find_node_in_meshes(const Node* n, const std::vector<Mesh*>& meshes)
{
    for (int m = 0; m < meshes.size(); ++m)
    {
        const std::vector<Node*>& nodes = meshes[m]->nodes;

        if (n->index < nodes.size() && n == nodes[n->index])
        {
            return m;
        }
    }
    
    return -1;
}

int find_edge_in_meshes(const Edge* e, const std::vector<Mesh*>& meshes)
{
    for (int m = 0; m < meshes.size(); ++m)
    {
        const std::vector<Edge*>& edges = meshes[m]->edges;

        if (e->index < edges.size() && e == edges[e->index])
        {
            return m;
        }
    }

    return -1;
}

int find_face_in_meshes(const Face* f, const std::vector<Mesh*>& meshes)
{
    for (int m = 0; m < meshes.size(); ++m)
    {
        const std::vector<Face*>& faces = meshes[m]->faces;

        if (f->index < faces.size() && f == faces[f->index])
        {
            return m;
        }
    }

    return -1;
}

void update_accel_struct(AccelStruct& acc) 
{
    // Refit Bounding Volumn Tree

    if (acc.root)
    {
        acc.tree.refit();
    }
}

void destroy_accel_structs(std::vector<AccelStruct*>& accs) 
{
    for (int a = 0; a < accs.size(); a++)
    {
        delete accs[a];
    }
}
