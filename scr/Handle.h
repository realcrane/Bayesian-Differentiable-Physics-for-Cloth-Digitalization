#pragma once

#include "Constraint.h"

struct Handle {
    Tensor start_time, end_time, fade_time;

    virtual ~Handle() {};
    virtual std::vector<Constraint*> get_constraints(Tensor t) = 0;
    virtual std::vector<Node*> get_nodes() = 0;
    bool active(Tensor t) { return (t >= start_time).item<int>() && (t <= end_time).item<int>(); }
    Tensor strength(Tensor t) {
        //if ((t < start_time).item<int>() || (t > end_time + fade_time).item<int>()) return ZERO;
        //if ((t <= end_time).item<int>()) return ONE;
        //Tensor s = 1 - (t - end_time) / (fade_time + 1e-6);
        //return s * s * s * s;

        return ONE;
    }
};

struct NodeHandle : public Handle {
    Node* node;
    //const Motion* motion;
    //Tensor velocity;
    bool activated;
    Tensor x0;
    NodeHandle() : activated(false) {}
    std::vector<Constraint*> get_constraints(Tensor t);
    std::vector<Node*> get_nodes() { return std::vector<Node*>(1, node); }
};

struct CircleHandle : public Handle {
    Mesh* mesh;
    int label;
    //const Motion* motion;
    Tensor c; // circumference
    Tensor u;
    Tensor xc, dx0, dx1;
    std::vector<Constraint*> get_constraints(Tensor t);
    std::vector<Node*> get_nodes() { return std::vector<Node*>(); }
};

struct GlueHandle : public Handle {
    Node* nodes[2];
    std::vector<Constraint*> get_constraints(Tensor t);
    std::vector<Node*> get_nodes() {
        std::vector<Node*> ns;
        ns.push_back(nodes[0]);
        ns.push_back(nodes[1]);
        return ns;
    }
};