#include "Handle.h"

static Tensor directions[3] = {
	torch::tensor({1,0,0},TNOPT), 
	torch::tensor({0,1,0},TNOPT), 
	torch::tensor({0,0,1},TNOPT) };

void add_position_constraints(const Node* node, const Tensor& x, Tensor stiff, std::vector<Constraint*>& cons) 
{
    for (int i = 0; i < 3; i++) {
        EqCon* con = new EqCon;
        con->node = (Node*)node;
        con->x = x;
        con->n = directions[i];
        con->stiff = stiff;
        cons.push_back(con);
    }
}

std::vector<Constraint*> NodeHandle::get_constraints(Tensor t) 
{
    
    Tensor s = strength(t);
    if ((s == 0).item<int>())
        return std::vector<Constraint*>();
    if (!activated) {
        x0 = node->x;
        activated = true;
    }

    Tensor x = x0;
   
    std::vector<Constraint*> cons;
    add_position_constraints(node, x, s * magic.handle_stiffness, cons);
    return cons;
}

std::vector<Constraint*> CircleHandle::get_constraints(Tensor t) {
    Tensor s = strength(t);
    if ((s == 0).item<int>())
        return std::vector<Constraint*>();
    std::vector<Constraint*> cons;
    for (int n = 0; n < mesh->nodes.size(); n++) {
        Node* node = mesh->nodes[n];
        if (node->label != label)
            continue;
        Tensor theta = 2 * M_PI * dot(node->verts[0]->u, u) / c;
        Tensor x = xc + (dx0 * cos(theta) + dx1 * sin(theta)) * c / (2 * M_PI);
        Tensor l = ZERO;
        for (int e = 0; e < node->adj_egdes.size(); e++) {
            const Edge* edge = node->adj_egdes[e];
            if (edge->nodes[0]->label != label || edge->nodes[1]->label != label)
                continue;
            l = l + edge->l;
        }
        add_position_constraints(node, x, s * magic.handle_stiffness * l, cons);
    }
    return cons;
}

std::vector<Constraint*> GlueHandle::get_constraints(Tensor t) {
    Tensor s = strength(t);
    if ((s == 0).item<int>())
        return std::vector<Constraint*>();
    std::vector<Constraint*> cons;
    for (int i = 0; i < 3; i++) {
        GlueCon* con = new GlueCon;
        con->nodes[0] = nodes[0];
        con->nodes[1] = nodes[1];
        con->n = directions[i];
        con->stiff = s * magic.handle_stiffness;
        cons.push_back(con);
    }
    return cons;
}
