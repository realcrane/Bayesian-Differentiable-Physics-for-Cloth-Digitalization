#pragma once

#include "Cloth.h"
#include "CollisionUtil.h"

#include "optimization.h"

using namespace alglib;

struct Impact {
    enum Type { VF, EE } type;
    Tensor t;
    Node* nodes[4];
    Tensor w[4];
    Tensor n;
    Impact() {}
    Impact(Type type, const Node* n0, const Node* n1, const Node* n2,
        const Node* n3) : type(type) {
        nodes[0] = (Node*)n0;
        nodes[1] = (Node*)n1;
        nodes[2] = (Node*)n2;
        nodes[3] = (Node*)n3;
    }
};

struct ImpactZone {
    std::vector<Node*> nodes;
    std::vector<Impact> impacts;
    std::vector<double> w, n;
    bool active;
};

std::vector<Tensor> apply_inelastic_projection_forward(Tensor xold, Tensor ws, Tensor ns, ImpactZone* zone);

std::vector<Tensor> apply_inelastic_projection_backward(Tensor dldx_tn, Tensor ans_tn, Tensor q_tn, Tensor r_tn, Tensor lam_tn, Tensor sm1_tn, Tensor legals_tn, ImpactZone* zone);

class InelasticProjection :public torch::autograd::Function<InelasticProjection> {
public:

    static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor xold, Tensor ws, Tensor ns, ImpactZone* zone) {

        std::vector<Tensor> ans = apply_inelastic_projection_forward(xold, ws, ns, zone);

        ctx->saved_data["zone_ptr"] = reinterpret_cast<intptr_t>(zone);

        ctx->save_for_backward(ans);

        return ans[0];
    }

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list dldz) {
        //std::cout << "Backward Called" << std::endl;
        auto saved = ctx->get_saved_variables();
        auto ans_tn = saved[0];
        auto q_tn = saved[1];
        auto r_tn = saved[2];
        auto lam_tn = saved[3];
        auto sm1_tn = saved[4];
        auto legals_tn = saved[5];

        ImpactZone* zone = reinterpret_cast<ImpactZone*>(ctx->saved_data["zone_ptr"].toInt());

        //ImpactZone* zone = static_cast<ImpactZone*>(const_cast<void*>(ctx->saved_data["zone"].internalToPointer()));
        auto ans_back = apply_inelastic_projection_backward(dldz[0], ans_tn, q_tn, r_tn, lam_tn, sm1_tn, legals_tn, zone);

        return { ans_back[0], ans_back[1], ans_back[2], torch::empty({}).grad() };

    }
};

struct Collision
{
    
    std::vector<Impact>* impacts = nullptr;

    int nthreads = 0;

    std::vector<std::pair<Face const*, Face const*>>* faceimpacts = nullptr;

    int* cnt = nullptr;

    void cloth_node_position();

    void obs_node_position();
    
    void collision_response(std::vector<Mesh*>& meshes, const std::vector<Mesh*>& obs_meshes);

    void update_active(const std::vector<AccelStruct*>& accs, const std::vector<AccelStruct*>& obs_accs, const std::vector<ImpactZone*>& zones);

    std::vector<Impact> find_impacts(const std::vector<AccelStruct*>& accs, const std::vector<AccelStruct*>& obs_accs);

    void find_face_impacts(const Face* face0, const Face* face1);

    void for_overlapping_faces(BVHNode* node, double thickness);

    void for_overlapping_faces(BVHNode* node0, BVHNode* node1, double thickness);

    void for_overlapping_faces(const std::vector<AccelStruct*>& accs, const std::vector<AccelStruct*>& obs_accs, Tensor thickness, bool parallel = true);

    void compute_face_impacts(const Face* face0, const Face* face1);

    bool vf_collision_test(const Vert* vert, const Face* face, Impact& impact);

    bool ee_collision_test(const Edge* edge0, const Edge* edge1, Impact& impact);

    bool collision_test(Impact::Type type, const Node* node0, const Node* node1, const Node* node2, const Node* node3, Impact& impact);

    void contact_jacobian(Impact& impact, Node* node);

    inline Tensor stp(const Tensor& u, const Tensor& v, const Tensor& w)
    {
        return torch::dot(u, torch::cross(v, w));
    }

    std::vector<Impact> independent_impacts(const std::vector<Impact>& impacts);

    bool conflict(const Impact& i0, const Impact& i1);

    void exclude(const ImpactZone* z, std::vector<ImpactZone*>& zs);

    int find_zone_in_zones(const ImpactZone* z, std::vector<ImpactZone*> zs);

    void remove_zone_from_zones(int i, std::vector<ImpactZone*>& zs);

    void add_impacts(const std::vector<Impact>& impacts, std::vector<ImpactZone*>& zones);

    ImpactZone* find_or_create_zone(const Node* node, std::vector<ImpactZone*>& zones);
    
    void merge_zones(ImpactZone* zone0, ImpactZone* zone1, std::vector<ImpactZone*>& zones);

    void apply_inelastic_projection(ImpactZone* zone, bool verbose);

};