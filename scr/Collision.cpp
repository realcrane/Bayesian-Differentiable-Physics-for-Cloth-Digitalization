#include <algorithm>
#include <omp.h>
#include <chrono>
// #include <execution>

#include "Collision.h"
#include "Solve.h"
#include "Geometry.h"
#include "Opti.h"

using namespace std::chrono;

static const Tensor& thickness = ::magic.projection_thickness;

std::vector<Mesh*> meshes, obs_meshes;

static std::vector<Tensor> xold;
static std::vector<Tensor> xold_obs;

inline int NEXT(int i)
{
    return (i < 2) ? (i + 1) : (i - 2);
}

inline int PREV(int i)
{
    return (i > 0) ? (i - 1) : (i + 2);
}

std::pair<bool, int> is_node_in_meshes(const Node* node) 
{
    int m = find_node_in_meshes(node, ::meshes);

    if (m != -1)
    {
        return std::make_pair(true, m);
    }
    else
    {
        return std::make_pair(false, find_node_in_meshes(node, ::obs_meshes));
    }
}

int find_node_in_nodes(const Node* n, Node* const* ns, int num_ns)
{
    for (int i = 0; i < num_ns; ++i)
    {
        if (ns[i] == n)
            return i;
    }

    return -1;
}

bool is_node_in_nodes(const Node* n, Node* const* ns, int num_ns)
{
    return find_node_in_nodes(n, ns, num_ns) != -1;
}

int find_node_in_nodes(const Node* n, std::vector<Node*>& ns)
{
    for (int i = 0; i < ns.size(); ++i)
    {
        if (ns[i] == n)
            return i;
    }
    
    return -1;
}

bool is_node_in_nodes(const Node* n, std::vector<Node*>& ns)
{
    return find_node_in_nodes(n, ns) != -1;
}

bool is_movable(const Node* n)
{
    // Return is the given node is in the meshes of cloths
    return find_node_in_meshes(n, ::meshes) != -1;
}

void Collision::exclude(const ImpactZone* z, std::vector<ImpactZone*>& zs)
{
    int i = find_zone_in_zones(z, zs);

    remove_zone_from_zones(i, zs);
}

void Collision::remove_zone_from_zones(int i, std::vector<ImpactZone*>& zs)
{
    zs[i] = zs.back();
    zs.pop_back();
}

int Collision::find_zone_in_zones(const ImpactZone* z, std::vector<ImpactZone*> zs)
{
    for (int i = 0; i < zs.size(); ++i)
    {
        if (zs[i] == z)
            return i;
    }

    return -1;
}

void Collision::cloth_node_position()
{
    xold.clear();

    for (int m = 0; m < ::meshes.size(); ++m) 
    {
        for (int n = 0; n < ::meshes[m]->nodes.size(); ++n) {
            xold.push_back(::meshes[m]->nodes[n]->x);
        }
    }
}

void Collision::obs_node_position()
{
    xold_obs.clear();

    for (int m = 0; m < ::obs_meshes.size(); ++m)
    {
        for (int n = 0; n < ::obs_meshes[m]->nodes.size(); ++n)
        {
            xold_obs.push_back(::obs_meshes[m]->nodes[n]->x);
        }
    }
}


void Collision::collision_response(std::vector<Mesh*>& meshes, const std::vector<Mesh*>& obs_meshes)
{
	
    //std::cout << "In Collision Response" << std::endl;

    static int max_iter = 100;

    ::meshes = meshes;
    ::obs_meshes = obs_meshes;

    cloth_node_position();  // Get Cloth Nodes' old position
    obs_node_position();    // Get Obstacle Nodes' old position

    bool verbose = false;

	std::vector<AccelStruct*> accs = create_accel_structs(::meshes, true);
	std::vector<AccelStruct*> obs_accs = create_accel_structs(::obs_meshes, true);

	std::vector<ImpactZone*> zones, prezones;

	int iter;
	static bool changed = false;
	static int count_changed = 0;
	static int num_step = 0;
	num_step++;

	zones.clear(); 
	prezones.clear();

    for (iter = 0; iter < max_iter; iter++)
    {        
        zones.clear();  // Clear Impace t Zone

        for (auto p : prezones) 
        {
            ImpactZone* newp = new ImpactZone;
            *newp = *p;
            zones.push_back(newp);
        }

        for (auto p : prezones)
            if (!p->active)
                delete p;

        if (!zones.empty())
            update_active(accs, obs_accs, zones);

        std::vector<Impact> impacts = find_impacts(accs, obs_accs);

        impacts = independent_impacts(impacts);

        if (impacts.empty())
            break;

        add_impacts(impacts, zones);

#pragma omp parallel for
        for (int z = 0; z < zones.size(); z++) {
            ImpactZone* zone = zones[z];
            if (zone->active) {
                changed = true;

                apply_inelastic_projection(zone, verbose);

            }
        }

        for (int a = 0; a < accs.size(); a++)
            update_accel_struct(*accs[a]);
        for (int a = 0; a < obs_accs.size(); a++)
            update_accel_struct(*obs_accs[a]);

        prezones = zones;
        count_changed++;
    }

    if (iter == max_iter) {
        std::cerr << "Collision resolution failed to converge!" << std::endl;
        exit(1);
    }


    for (int z = 0; z < zones.size(); z++)
    {
        delete zones[z];
    }

    destroy_accel_structs(accs);
    destroy_accel_structs(obs_accs);
}

void Collision::update_active(const std::vector<AccelStruct*>& accs, const std::vector<AccelStruct*>& obs_accs, const std::vector<ImpactZone*>& zones) 
{
    
    for (int a = 0; a < accs.size(); a++)
    {
        mark_all_inactive(*accs[a]);
    }

    for (int z = 0; z < zones.size(); z++) 
    {
        const ImpactZone* zone = zones[z];
        if (!zone->active)
            continue;
        for (int n = 0; n < zone->nodes.size(); n++) 
        {
            const Node* node = zone->nodes[n];
            std::pair<bool, int> mi = is_node_in_meshes(node);

            AccelStruct* acc = (mi.first ? accs : obs_accs)[mi.second];

            for (int v = 0; v < node->verts.size(); v++)
                for (int f = 0; f < node->verts[v]->adj_faces.size(); f++)
                    mark_active(*acc, node->verts[v]->adj_faces[f]);
        }
    }
}

void Collision::find_face_impacts(const Face* face0, const Face* face1)
{
    // Callback function

    int t = omp_get_thread_num();

    faceimpacts[t].push_back(std::make_pair(face0, face1));
}

std::vector<Impact> Collision::find_impacts(const std::vector<AccelStruct*>& accs, const std::vector<AccelStruct*>& obs_accs)
{
    if (impacts == nullptr) 
    {
        nthreads = omp_get_max_threads();
        impacts = new std::vector<Impact>[nthreads];
        faceimpacts = new std::vector<std::pair<Face const*, Face const*>> [nthreads];
        cnt = new int[nthreads];
    }

    for (int t = 0; t < nthreads; t++) 
    {
        impacts[t].clear();
        faceimpacts[t].clear();
        cnt[t] = 0;
    }

    for_overlapping_faces(accs, obs_accs, thickness);

    std::vector<std::pair<Face const*, Face const*>> tot_faces;

    for (int t = 0; t < nthreads; ++t) 
    {
        tot_faces.insert(tot_faces.end(), faceimpacts[t].begin(), faceimpacts[t].end());
    }

#pragma omp parallel for
    for (int i = 0; i < tot_faces.size(); ++i) {
        compute_face_impacts(tot_faces[i].first, tot_faces[i].second);
    }

    std::vector<Impact> loc_impacts;

    for (int t = 0; t < nthreads; t++) {
        loc_impacts.insert(loc_impacts.end(), impacts[t].begin(), impacts[t].end());
    }

    return loc_impacts;
}


void Collision::compute_face_impacts(const Face* face0, const Face* face1)
{
    int t = omp_get_thread_num();

    Impact impact;

    kDOP18 nb[6], eb[6], fb[2];

    for (int v = 0; v < 3; ++v) 
    {
        nb[v] = node_box(face0->v[v]->node, true);
        nb[v + 3] = node_box(face1->v[v]->node, true);
    }

    for (int v = 0; v < 3; ++v) 
    {
        eb[v] = nb[NEXT(v)] + nb[PREV(v)];
        eb[v + 3] = nb[NEXT(v) + 3] + nb[PREV(v) + 3];
    }

    fb[0] = nb[0] + nb[1] + nb[2];
    fb[1] = nb[3] + nb[4] + nb[5];

    double thick = thickness.item<double>();

    for (int v = 0; v < 3; v++) {
        if (!overlap(nb[v], fb[1], thick))
            continue;
        if (vf_collision_test(face0->v[v], face1, impact))
            impacts[t].push_back(impact);
    }

    for (int v = 0; v < 3; v++) {
        if (!overlap(nb[v + 3], fb[0], thick))
            continue;
        if (vf_collision_test(face1->v[v], face0, impact))
            impacts[t].push_back(impact);
    }

    for (int e0 = 0; e0 < 3; e0++) {
        for (int e1 = 0; e1 < 3; e1++)
        {
            if (!overlap(eb[e0], eb[e1 + 3], thick))
                continue;
            if (ee_collision_test(face0->adj_edges[e0], face1->adj_edges[e1], impact))
                impacts[t].push_back(impact);
        }
    }
}

bool Collision::vf_collision_test(const Vert* vert, const Face* face, Impact& impact) 
{
    const Node* node = vert->node;
    if (node == face->v[0]->node
        || node == face->v[1]->node
        || node == face->v[2]->node)
        return false;

    return collision_test(Impact::VF, node, face->v[0]->node, face->v[1]->node, face->v[2]->node, impact);
}

bool Collision::ee_collision_test(const Edge* edge0, const Edge* edge1, Impact& impact) {
    if (edge0->nodes[0] == edge1->nodes[0] || edge0->nodes[0] == edge1->nodes[1]
        || edge0->nodes[1] == edge1->nodes[0] || edge0->nodes[1] == edge1->nodes[1])
        return false;

    return collision_test(Impact::EE, edge0->nodes[0], edge0->nodes[1], edge1->nodes[0], edge1->nodes[1], impact);
}


void Collision::contact_jacobian(Impact& impact, Node* node) 
{

}

bool Collision::collision_test(Impact::Type type, const Node* node0, const Node* node1, const Node* node2, const Node* node3, Impact& impact) 
{
    int t0 = omp_get_thread_num();
    ++cnt[t0];
    impact.type = type;

    const Tensor& x0 = node0->x0, v0 = node0->x - x0;
    Tensor x1 = node1->x0 - x0, x2 = node2->x0 - x0, x3 = node3->x0 - x0;
    Tensor v1 = (node1->x - node1->x0) - v0, v2 = (node2->x - node2->x0) - v0,
        v3 = (node3->x - node3->x0) - v0;

    Tensor a0 = stp(x1, x2, x3);
    Tensor a1 = stp(v1, x2, x3) + stp(x1, v2, x3) + stp(x1, x2, v3);
    Tensor a2 = stp(x1, v2, v3) + stp(v1, x2, v3) + stp(v1, v2, x3);
    Tensor a3 = stp(v1, v2, v3);

    //Tensor t = solve_cubic_forward(a3, a2, a1, a0).detach();
    Tensor t = CubicSolver::apply(a3, a2, a1, a0).detach();
    int nsol = t.size(0);
    
    for (int i = 0; i < nsol; i++) {
        if ((t[i] < 0).item<int>() || (t[i] > 1).item<int>())
            continue;
        impact.t = t[i];
        Tensor bx0 = x0 + t[i] * v0, bx1 = x1 + t[i] * v1,
            bx2 = x2 + t[i] * v2, bx3 = x3 + t[i] * v3;
        Tensor& n = impact.n;
        Tensor* w = impact.w;
        w[0] = w[1] = w[2] = w[3] = ZERO;
        Tensor d;
        bool inside, over = false;
        if (type == Impact::VF) {
            d = sub_signed_vf_distance(bx1, bx2, bx3, &n, w, 1e-6, over);
            inside = (torch::min(-w[1], torch::min(-w[2], -w[3])) >= -1e-6).item<int>();
        }
        else {
            d = sub_signed_ee_distance(bx1, bx2, bx3, bx2 - bx1, bx3 - bx1, bx3 - bx2, &n, w, 1e-6, over);
            inside = (torch::min(torch::min(w[0], w[1]), torch::min(-w[2], -w[3])) >= -1e-6).item<int>();
        }
        if (over || !inside)
            continue;
        if ((torch::dot(n, w[1] * v1 + w[2] * v2 + w[3] * v3) > 0).item<int>())
            n = -n;

        impact.nodes[0] = const_cast<Node*>(node0);
        impact.nodes[1] = const_cast<Node*>(node1);
        impact.nodes[2] = const_cast<Node*>(node2);
        impact.nodes[3] = const_cast<Node*>(node3);

        return true;
    }
    return false;
}

void Collision::for_overlapping_faces(BVHNode* node, double thickness)
{
    if (node->isLeaf() || !node->_active)
        return;
    for_overlapping_faces(node->getLeftChild(), thickness);
    for_overlapping_faces(node->getRightChild(), thickness);
    for_overlapping_faces(node->getLeftChild(), node->getRightChild(), thickness);
}

void Collision::for_overlapping_faces(BVHNode* node0, BVHNode* node1, double thickness)
{
    if (!node0->_active && !node1->_active)
        return;
    if (!overlap(node0->_box, node1->_box, thickness))
        return;
    if (node0->isLeaf() && node1->isLeaf()) {
        Face* face0 = node0->getFace(),
            * face1 = node1->getFace();
        find_face_impacts(face0, face1);
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

std::vector<BVHNode*> collect_upper_nodes(const std::vector<AccelStruct*>& accs, int n);

void Collision::for_overlapping_faces(const std::vector<AccelStruct*>& accs, const std::vector<AccelStruct*>& obs_accs, Tensor thickness0, bool parallel)
{
    //omp_set_num_threads(1);
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
        //for (int o = 0; o < obs_accs.size(); o++)
        //    if (obs_accs[o]->root)
        //        for_overlapping_faces(nodes[n], obs_accs[o]->root, thickness,
        //                              callback);
    }
    omp_set_num_threads(nthreads);
}

std::vector<BVHNode*> collect_upper_nodes(const std::vector<AccelStruct*>& accs, int nnodes) {

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

bool operator< (const Impact& impact0, const Impact& impact1) 
{
    return (impact0.t < impact1.t).item<bool>();
}


std::vector<Impact> Collision::independent_impacts(const std::vector<Impact>& impacts) 
{
    std::vector<Impact> sorted = impacts;
    std::sort(sorted.begin(), sorted.end());

    std::vector<Impact> indep;

    for (int e = 0; e < sorted.size(); e++) {
        const Impact& impact = sorted[e];

        bool con = false;
        for (int e1 = 0; e1 < indep.size(); e1++)
            if (conflict(impact, indep[e1])) {
                con = true;
                break;
            }               
        if (!con)
            indep.push_back(impact);
    }

    return indep;
}

bool Collision::conflict(const Impact& i0, const Impact& i1) 
{
    return (is_movable(i0.nodes[0]) && is_node_in_nodes(i0.nodes[0], i1.nodes, 4))
        || (is_movable(i0.nodes[1]) && is_node_in_nodes(i0.nodes[1], i1.nodes, 4))
        || (is_movable(i0.nodes[2]) && is_node_in_nodes(i0.nodes[2], i1.nodes, 4))
        || (is_movable(i0.nodes[3]) && is_node_in_nodes(i0.nodes[3], i1.nodes, 4));
}

void Collision::add_impacts(const std::vector<Impact>& impacts, std::vector<ImpactZone*>& zones) 
{
   
    for (int z = 0; z < zones.size(); z++) {
        zones[z]->active = false;
    }

    for (int i = 0; i < impacts.size(); i++) 
    {        
        const Impact& impact = impacts[i];

        Node* node = impact.nodes[is_movable(impact.nodes[0]) ? 0 : 3];

        ImpactZone* zone = find_or_create_zone(node, zones);

        for (int n = 0; n < 4; n++) {
            if (is_movable(impact.nodes[n])) {               
                merge_zones(zone, find_or_create_zone(impact.nodes[n], zones), zones);
            }
        }

        zone->impacts.push_back(impact);
        zone->active = true;
    }
}

ImpactZone* Collision::find_or_create_zone(const Node* node, std::vector<ImpactZone*>& zones) 
{
    for (int z = 0; z < zones.size(); z++) {
        if (find_node_in_nodes(node, zones[z]->nodes) != -1) {                       
            return zones[z];
        }
    }

    ImpactZone* zone = new ImpactZone;

    zone->nodes.push_back(const_cast<Node*>(node));

    zones.push_back(zone);

    return zone;
}

void Collision::merge_zones(ImpactZone* zone0, ImpactZone* zone1, std::vector<ImpactZone*>& zones) 
{
    if (zone0 == zone1) {
        return;
    }
    zone0->nodes.insert(zone0->nodes.end(), zone1->nodes.begin(), zone1->nodes.end());
    zone0->impacts.insert(zone0->impacts.end(), zone1->impacts.begin(), zone1->impacts.end());
    exclude(zone1, zones);
    delete zone1;
}

int get_index(const Node* node, bool is_cloth)
{
    int i = 0;

    if (is_cloth)
    {
        for (int m = 0; m < ::meshes.size(); ++m)
        {
            const std::vector<Node*>& ns = ::meshes[m]->nodes;
            if (node->index < ns.size() && node == ns[node->index])
                return i + node->index;
            else
                i += ns.size();
        }
    }
    else
    {
        for (int m = 0; m < ::obs_meshes.size(); ++m)
        {
            const std::vector<Node*>& ns = ::obs_meshes[m]->nodes;
            if (node->index < ns.size() && node == ns[node->index])
                return i + node->index;
            else
                i += ns.size();
        }
    }

    return -1;
}

Tensor& get_xold(const Node* node)
{
    std::pair<bool, int> mi = is_node_in_meshes(node);

    int ni = get_index(node, mi.first);

    return (mi.first ? xold : xold_obs)[ni];
}

Tensor get_mass(const Node* node)
{
    return node->m;
}

struct NormalOpt : public NLConOpt {
    ImpactZone* zone;
    Tensor inv_m;
    std::vector<double> tmp;
    NormalOpt() : zone(NULL), inv_m(ZERO) { nvar = ncon = 0; }
    NormalOpt(ImpactZone* zone) : zone(zone), inv_m(ZERO) 
    {
        nvar = zone->nodes.size() * 3;
        ncon = zone->impacts.size();
        for (int n = 0; n < zone->nodes.size(); n++)
            inv_m = inv_m + 1 / get_mass(zone->nodes[n]);         
        inv_m = inv_m / (double)zone->nodes.size();
        tmp = std::vector<double>(nvar);
    }
    void initialize(double* x) const;
    void precompute(const double* x) const;
    double objective(const double* x) const;
    void obj_grad(const double* x, double* grad) const;
    double constraint(const double* x, int i, int& sign) const;
    void con_grad(const double* x, int i, double factor, double* grad) const;
    //void con_grad(const double* x, int i, double factor, std::vector<double>& grad) const;
    void finalize(const double* x);
};

void precompute_derivative(real_2d_array& a, real_2d_array& q, real_2d_array& r0, std::vector<double>& lambda, real_1d_array& sm_1, std::vector<int>& legals, double** grads, ImpactZone* zone, NormalOpt& slx)
{
    a.setlength(slx.nvar, legals.size());
    sm_1.setlength(slx.nvar);
    for (int i = 0; i < slx.nvar; ++i)
        sm_1[i] = 1.0 / sqrt(slx.inv_m * get_mass(zone->nodes[i / 3])).item<double>();
    for (int k = 0; k < legals.size(); ++k)
        for (int i = 0; i < slx.nvar; ++i)
            a[i][k] = grads[legals[k]][i] * sm_1[i]; //sqrt(m^-1)
    real_1d_array tau, r1lam1, lamp;
    tau.setlength(slx.nvar);
    rmatrixqr(a, slx.nvar, legals.size(), tau);
    real_2d_array qtmp, r, r1;
    int cols = legals.size();
    if (cols > slx.nvar)cols = slx.nvar;
    rmatrixqrunpackq(a, slx.nvar, legals.size(), tau, cols, qtmp);
    rmatrixqrunpackr(a, slx.nvar, legals.size(), r);
    // get rid of degenerate G
    int newdim = 0;
    for (; newdim < cols; ++newdim)
        if (abs(r[newdim][newdim]) < 1e-6)
            break;
    r0.setlength(newdim, newdim);
    r1.setlength(newdim, legals.size() - newdim);
    q.setlength(slx.nvar, newdim);
    for (int i = 0; i < slx.nvar; ++i)
        for (int j = 0; j < newdim; ++j)
            q[i][j] = qtmp[i][j];
    for (int i = 0; i < newdim; ++i) {
        for (int j = 0; j < newdim; ++j)
            r0[i][j] = r[i][j];
        for (int j = newdim; j < legals.size(); ++j)
            r1[i][j - newdim] = r[i][j];
    }
    r1lam1.setlength(newdim);
    for (int i = 0; i < newdim; ++i) {
        r1lam1[i] = 0;
        for (int j = newdim; j < legals.size(); ++j)
            r1lam1[i] += r1[i][j - newdim] * lambda[legals[j]];
    }
    ae_int_t info;
    alglib::densesolverreport rep;
    rmatrixsolve(r0, (ae_int_t)newdim, r1lam1, info, rep, lamp);
    for (int j = 0; j < newdim; ++j)
        lambda[legals[j]] += lamp[j];
    for (int j = newdim; j < legals.size(); ++j)
        lambda[legals[j]] = 0;
}

std::vector<double> ten2vecd(Tensor a) {
    std::vector<double> ans;
    double* x = a.data<double>();
    int n = a.size(0);
    for (int i = 0; i < n; ++i)
        ans.push_back(x[i]);
    return ans;
}

std::vector<int> ten2veci(Tensor a) {
    std::vector<int> ans;
    int* x = a.data<int>();
    int n = a.size(0);
    for (int i = 0; i < n; ++i)
        ans.push_back(x[i]);
    return ans;
}

Tensor arr2ten(real_2d_array a) {
    int n = a.rows(), m = a.cols();
    std::vector<double> tmp;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            tmp.push_back(a[i][j]);
    Tensor ans = torch::tensor(tmp, TNOPT).reshape({ n, m });
    return ans;
}

real_2d_array ten2arr(Tensor a) {
    int n = a.size(0), m = a.size(1);
    auto foo_a = a.accessor<double, 2>();
    real_2d_array ans;
    ans.setlength(n, m);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            ans[i][j] = foo_a[i][j];
    return ans;
}

real_1d_array ten1arr(Tensor a) {
    int n = a.size(0);
    auto foo_a = a.accessor<double, 1>();
    real_1d_array ans;
    ans.setlength(n);
    for (int i = 0; i < n; ++i)
        ans[i] = foo_a[i];
    return ans;
}

Tensor ptr2ten(double* a, int n) {
    std::vector<double> b;
    for (int i = 0; i < n; ++i)
        b.push_back(a[i]);
    return torch::tensor(b, TNOPT);
}

Tensor ptr2ten(int* a, int n) {
    std::vector<int> b;
    for (int i = 0; i < n; ++i)
        b.push_back(a[i]);
    return torch::tensor(b, torch::dtype(torch::kI32));
}

inline Tensor get_subvec(const double* x, int i) {
    return torch::tensor({ x[i * 3 + 0], x[i * 3 + 1], x[i * 3 + 2] }, TNOPT);
}
inline void set_subvec(double* x, int i, const Tensor& xi) {
    for (int j = 0; j < 3; j++) x[i * 3 + j] = xi[j].item<double>();
}
inline void add_subvec(double* x, int i, const Tensor& xi) {
    for (int j = 0; j < 3; j++) x[i * 3 + j] += xi[j].item<double>();
}

inline Tensor get_subvec(const Tensor x, int i) {
    return x.slice(0, i * 3, i * 3 + 3);
}
inline void set_subvec(Tensor& x, int i, const Tensor& xi) {
    x.slice(0, i * 3, i * 3 + 3) = xi;
}
inline void add_subvec(Tensor& x, int i, const Tensor& xi) {
    x.slice(0, i * 3, i * 3 + 3) += xi;
}

std::vector<Tensor> apply_inelastic_projection_forward(Tensor xold, Tensor ws, Tensor ns, ImpactZone* zone)
{
    auto slx = NormalOpt(zone);

    double* x = new double[slx.nvar];
    double* oricon = new double[slx.ncon];
    int sign;
    slx.initialize(x);
    auto lambda = augmented_lagrangian_method(slx);
    // do qr decomposition on sqrt(m^-1)G^T
    std::vector<int> legals;
    double** grads = new double* [slx.ncon];
    double tmp;
    for (int i = 0; i < slx.ncon; ++i) {
        tmp = slx.constraint(&slx.tmp[0], i, sign);
        grads[i] = NULL;
        if (sign == 1 && tmp > 1e-6) continue;//sign==1:tmp>=0
        if (sign == -1 && tmp < -1e-6) continue;
        grads[i] = new double[slx.nvar];
        for (int j = 0; j < slx.nvar; ++j)
            grads[i][j] = 0;
        slx.con_grad(&slx.tmp[0], i, 1, grads[i]);
        legals.push_back(i);
    }
    real_2d_array a, q, r;
    real_1d_array sm_1;//sqrt(m^-1)
    precompute_derivative(a, q, r, lambda, sm_1, legals, grads, zone, slx);
    Tensor q_tn = arr2ten(q), r_tn = arr2ten(r);
    Tensor lam_tn = ptr2ten(&lambda[0], lambda.size());
    Tensor sm1_tn = ptr2ten(sm_1.getcontent(), sm_1.length());
    Tensor legals_tn = ptr2ten(&legals[0], legals.size());
    Tensor ans = ptr2ten(&slx.tmp[0], slx.nvar);

    delete[] x;
    delete[] oricon;

    for (int i = 0; i < slx.ncon; ++i) {
        delete[] grads[i];
    }

    return { ans.reshape({-1, 3}), q_tn, r_tn, lam_tn, sm1_tn, legals_tn };
}

std::vector<Tensor> compute_derivative(real_1d_array& ans, ImpactZone* zone,
    real_2d_array& q, real_2d_array& r, real_1d_array& sm_1, std::vector<int>& legals,
    real_1d_array& dldx,
    std::vector<double>& lambda, bool verbose = false) {
    real_1d_array qtx, dz, dlam0, dlam, ana, dldw0, dldn0;
    int nvar = zone->nodes.size() * 3;
    int ncon = zone->impacts.size();
    qtx.setlength(q.cols());
    ana.setlength(nvar);
    dldn0.setlength(ncon * 3);
    dldw0.setlength(ncon * 4);
    dz.setlength(nvar);
    dlam0.setlength(q.cols());
    dlam.setlength(ncon);
    for (int i = 0; i < nvar; ++i)
        ana[i] = dz[i] = 0;
    for (int i = 0; i < ncon * 3; ++i) dldn0[i] = 0;
    for (int i = 0; i < ncon * 4; ++i) dldw0[i] = 0;
    // qtx = qt * sqrt(m^-1) dldx
    for (int i = 0; i < q.cols(); ++i) {
        qtx[i] = 0;
        for (int j = 0; j < nvar; ++j)
            qtx[i] += q[j][i] * dldx[j] * sm_1[j];
    }
    // dz = sqrt(m^-1) (sqrt(m^-1) dldx - q * qtx)
    for (int i = 0; i < nvar; ++i) {
        dz[i] = dldx[i] * sm_1[i];
        for (int j = 0; j < q.cols(); ++j)
            dz[i] -= q[i][j] * qtx[j];
        dz[i] *= sm_1[i];
    }
    // dlam = R^-1 * qtx
    ae_int_t info;
    alglib::densesolverreport rep;
    //std::cout << "orisize=" << nvar << " " << ncon << " " << nvar + ncon;
    //std::cout << "  size=" << q.cols() << std::endl;
    rmatrixsolve(r, (ae_int_t)q.cols(), qtx, info, rep, dlam0);
    // cout<<endl;
    for (int j = 0; j < ncon; ++j)
        dlam[j] = 0;
    for (int k = 0; k < q.cols(); ++k)
        dlam[legals[k]] = dlam0[k];
    //part1: dldq * dqdxt = M dz
    for (int i = 0; i < nvar; ++i)
        ana[i] += dz[i] / sm_1[i] / sm_1[i];
    //part2: dldg * dgdw * dwdxt
    for (int j = 0; j < ncon; ++j) {
        Impact& imp = zone->impacts[j];
        double* dldn = dldn0.getcontent() + j * 3;
        for (int n = 0; n < 4; n++) {
            int i = find_node_in_nodes(imp.nodes[n], zone->nodes);  // **replaced**
            double& dldw = dldw0[j * 4 + n];
            if (i != -1) {
                for (int k = 0; k < 3; ++k) {
                    //g=-w*n*x
                    dldw += (dlam[j] * ans[i * 3 + k] + lambda[j] * dz[i * 3 + k]) * imp.n[k].item<double>();
                    //part3: dldg * dgdn * dndxt
                    dldn[k] += imp.w[n].item<double>() * (dlam[j] * ans[i * 3 + k] + lambda[j] * dz[i * 3 + k]);
                }
            }
            else {
                //part4: dldh * (dhdw + dhdn)
                for (int k = 0; k < 3; ++k) {
                    dldw += (dlam[j] * imp.n[k] * imp.nodes[n]->x[k]).item<double>();
                    dldn[k] += (dlam[j] * imp.w[n] * imp.nodes[n]->x[k]).item<double>();
                }
            }
        }
    }
    Tensor grad_xold = torch::from_blob(ana.getcontent(), { nvar / 3, 3 }, TNOPT).clone();
    Tensor grad_w = torch::from_blob(dldw0.getcontent(), { ncon * 4 }, TNOPT).clone();
    Tensor grad_n = torch::from_blob(dldn0.getcontent(), { ncon, 3 }, TNOPT).clone();
    // cout << grad_xold<<endl << grad_w.reshape({ncon,4})<<endl << grad_n<<endl << endl;
    delete zone;
    return { grad_xold, grad_w, grad_n };
}

std::vector<Tensor> apply_inelastic_projection_backward(Tensor dldx_tn, Tensor ans_tn, Tensor q_tn, Tensor r_tn, Tensor lam_tn, Tensor sm1_tn, Tensor legals_tn, ImpactZone* zone) 
{
    real_2d_array q = ten2arr(q_tn), r = ten2arr(r_tn);
    real_1d_array sm_1 = ten1arr(sm1_tn), ans = ten1arr(ans_tn.reshape({ -1 })), dldx = ten1arr(dldx_tn.reshape({ -1 }));
    std::vector<double> lambda = ten2vecd(lam_tn);  // replaced
    std::vector<int> legals = ten2veci(legals_tn);  // replaced
    // cout << dldx_tn<<endl << q_tn<<endl;
    return compute_derivative(ans, zone, q, r, sm_1, legals, dldx, lambda);
}

void Collision::apply_inelastic_projection(ImpactZone* zone, bool verbose)
{
    Tensor inp_xold;
    Tensor inp_w;
    Tensor inp_n;

    std::vector<Tensor> xolds, ws, ns;

    for (int i = 0; i < zone->nodes.size(); ++i) {
        xolds.push_back(get_xold(zone->nodes[i]));
    }

    for (int j = 0; j < zone->impacts.size(); ++j) 
    {
        ns.push_back(zone->impacts[j].n);

        for (int k = 0; k < 4; ++k) {
            ws.push_back(zone->impacts[j].w[k]);
        }
    }

    inp_xold = torch::stack(xolds);
    inp_w = torch::stack(ws);
    inp_n = torch::stack(ns);

    zone->w = std::vector<double>(inp_w.data_ptr<double>(), inp_w.data_ptr<double>() + inp_w.numel());
    zone->n = std::vector<double>(inp_n.data_ptr<double>(), inp_n.data_ptr<double>() + inp_n.numel());

    //Tensor out_x = apply_inelastic_projection_forward(inp_xold, inp_w, inp_n, zone)[0];
    Tensor out_x = InelasticProjection::apply(inp_xold, inp_w, inp_n, zone);

    for (int i = 0; i < zone->nodes.size(); ++i) {
        zone->nodes[i]->x = out_x[i];
    }
}

void NormalOpt::initialize(double* x) const {
    for (int n = 0; n < zone->nodes.size(); n++)
        set_subvec(x, n, zone->nodes[n]->x);
}

void NormalOpt::precompute(const double* x) const {
    for (int n = 0; n < zone->nodes.size(); n++)
        zone->nodes[n]->x = get_subvec(x, n);
}

double NormalOpt::objective(const double* x) const {
    double e = 0;
    for (int n = 0; n < zone->nodes.size(); n++) {
        const Node* node = zone->nodes[n];
        Tensor dx = node->x - get_xold(node);
        e = e + (inv_m * get_mass(node) * dot(dx, dx) / 2).item<double>();
    }
    return e;
}

void NormalOpt::obj_grad(const double* x, double* grad) const {
    for (int n = 0; n < zone->nodes.size(); n++) {
        const Node* node = zone->nodes[n];
        Tensor dx = node->x - get_xold(node);
        set_subvec(grad, n, inv_m * get_mass(node) * dx);
    }
}

double NormalOpt::constraint(const double* x, int j, int& sign) const {
    sign = -1;
    double c = ::thickness.item<double>();
    const Impact& impact = zone->impacts[j];
    for (int n = 0; n < 4; n++)
    {
        double* dx = impact.nodes[n]->x.data<double>();
        for (int k = 0; k < 3; ++k) {
            c -= zone->w[j * 4 + n] * zone->n[j * 3 + k] * dx[k];
        }
    }
    return c;
}


void NormalOpt::con_grad(const double* x, int j, double factor, double* grad) const
{
    const Impact& impact = zone->impacts[j];
    for (int n = 0; n < 4; n++) {
        int i = find_node_in_nodes(impact.nodes[n], zone->nodes);
        if (i != -1)
            for (int k = 0; k < 3; ++k)
            {
                grad[i * 3 + k] -= factor * zone->w[j * 4 + n] * zone->n[j * 3 + k];
            }
                
    }
}

void NormalOpt::finalize(const double* x) {
    precompute(x);
    for (int i = 0; i < nvar; ++i)
        tmp[i] = x[i];
}



