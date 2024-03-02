#pragma once
#include <float.h>
#include "Mesh.h"

using vec3f = std::vector<double>;

inline double MAX(double a, double b)
{
	return a > b ? a : b;
}

inline double MIN(double a, double b)
{
	return a < b ? a : b;
}

inline vec3f dat2vec3(double* x) 
{
	return std::vector<double>(x, x + 3);
}

class kDOP18 {
public:
	inline static void getDistances(const vec3f& p,
		double& d3, double& d4, double& d5, double& d6, double& d7, double& d8)
	{
		d3 = p[0] + p[1];
		d4 = p[0] + p[2];
		d5 = p[1] + p[2];
		d6 = p[0] - p[1];
		d7 = p[0] - p[2];
		d8 = p[1] - p[2];
	}

	inline static void getDistances(const vec3f& p, double d[])
	{
		d[0] = p[0] + p[1];
		d[1] = p[0] + p[2];
		d[2] = p[1] + p[2];
		d[3] = p[0] - p[1];
		d[4] = p[0] - p[2];
		d[5] = p[1] - p[2];
	}

	inline static double getDistances(const vec3f& p, int i)
	{
		if (i == 0) return p[0] + p[1];
		if (i == 1) return p[0] + p[2];
		if (i == 2) return p[1] + p[2];
		if (i == 3) return p[0] - p[1];
		if (i == 4) return p[0] - p[2];
		if (i == 5) return p[1] - p[2];
		return 0;
	}

public:
	double _dist[18];

	inline kDOP18() {
		empty();
	}

	inline kDOP18(const vec3f& v) {
		_dist[0] = _dist[9] = v[0];
		_dist[1] = _dist[10] = v[1];
		_dist[2] = _dist[11] = v[2];

		double d3, d4, d5, d6, d7, d8;
		getDistances(v, d3, d4, d5, d6, d7, d8);
		_dist[3] = _dist[12] = d3;
		_dist[4] = _dist[13] = d4;
		_dist[5] = _dist[14] = d5;
		_dist[6] = _dist[15] = d6;
		_dist[7] = _dist[16] = d7;
		_dist[8] = _dist[17] = d8;
	}

	inline kDOP18(const vec3f& a, const vec3f& b) {
		_dist[0] = MIN(a[0], b[0]);
		_dist[9] = MAX(a[0], b[0]);
		_dist[1] = MIN(a[1], b[1]);
		_dist[10] = MAX(a[1], b[1]);
		_dist[2] = MIN(a[2], b[2]);
		_dist[11] = MAX(a[2], b[2]);

		double ad3, ad4, ad5, ad6, ad7, ad8;
		getDistances(a, ad3, ad4, ad5, ad6, ad7, ad8);
		double bd3, bd4, bd5, bd6, bd7, bd8;
		getDistances(b, bd3, bd4, bd5, bd6, bd7, bd8);
		_dist[3] = MIN(ad3, bd3);
		_dist[12] = MAX(ad3, bd3);
		_dist[4] = MIN(ad4, bd4);
		_dist[13] = MAX(ad4, bd4);
		_dist[5] = MIN(ad5, bd5);
		_dist[14] = MAX(ad5, bd5);
		_dist[6] = MIN(ad6, bd6);
		_dist[15] = MAX(ad6, bd6);
		_dist[7] = MIN(ad7, bd7);
		_dist[16] = MAX(ad7, bd7);
		_dist[8] = MIN(ad8, bd8);
		_dist[17] = MAX(ad8, bd8);
	}

	inline bool overlaps(const kDOP18& b) const
	{
		for (int i = 0; i < 9; i++) {
			if (_dist[i] > b._dist[i + 9]) return false;
			if (_dist[i + 9] < b._dist[i]) return false;
		}

		return true;
	}

	inline bool overlaps(const kDOP18& b, kDOP18& ret) const
	{
		if (!overlaps(b))
			return false;

		for (int i = 0; i < 9; i++) {
			ret._dist[i] = MAX(_dist[i], b._dist[i]);
			ret._dist[i + 9] = MIN(_dist[i + 9], b._dist[i + 9]);
		}
		return true;
	}

	inline bool inside(const vec3f& p) const
	{
		for (int i = 0; i < 3; i++) {
			if (p[i] < _dist[i] || p[i] > _dist[i + 9])
				return false;
		}

		double d[6];
		getDistances(p, d);
		for (int i = 3; i < 9; i++) {
			if (d[i - 3] < _dist[i] || d[i - 3] > _dist[i + 9])
				return false;
		}

		return true;
	}

	inline kDOP18& operator += (const Tensor& p0)
	{
		vec3f p = dat2vec3(p0.data<double>());
		return *this += p;
	}
	inline kDOP18& operator += (const vec3f& p)
	{
		_dist[0] = MIN(p[0], _dist[0]);
		_dist[9] = MAX(p[0], _dist[9]);
		_dist[1] = MIN(p[1], _dist[1]);
		_dist[10] = MAX(p[1], _dist[10]);
		_dist[2] = MIN(p[2], _dist[2]);
		_dist[11] = MAX(p[2], _dist[11]);

		double d3, d4, d5, d6, d7, d8;
		getDistances(p, d3, d4, d5, d6, d7, d8);
		_dist[3] = MIN(d3, _dist[3]);
		_dist[12] = MAX(d3, _dist[12]);
		_dist[4] = MIN(d4, _dist[4]);
		_dist[13] = MAX(d4, _dist[13]);
		_dist[5] = MIN(d5, _dist[5]);
		_dist[14] = MAX(d5, _dist[14]);
		_dist[6] = MIN(d6, _dist[6]);
		_dist[15] = MAX(d6, _dist[15]);
		_dist[7] = MIN(d7, _dist[7]);
		_dist[16] = MAX(d7, _dist[16]);
		_dist[8] = MIN(d8, _dist[8]);
		_dist[17] = MAX(d8, _dist[17]);

		return *this;
	}

	inline kDOP18& operator += (const kDOP18& b)
	{
		_dist[0] = MIN(b._dist[0], _dist[0]);
		_dist[9] = MAX(b._dist[9], _dist[9]);
		_dist[1] = MIN(b._dist[1], _dist[1]);
		_dist[10] = MAX(b._dist[10], _dist[10]);
		_dist[2] = MIN(b._dist[2], _dist[2]);
		_dist[11] = MAX(b._dist[11], _dist[11]);
		_dist[3] = MIN(b._dist[3], _dist[3]);
		_dist[12] = MAX(b._dist[12], _dist[12]);
		_dist[4] = MIN(b._dist[4], _dist[4]);
		_dist[13] = MAX(b._dist[13], _dist[13]);
		_dist[5] = MIN(b._dist[5], _dist[5]);
		_dist[14] = MAX(b._dist[14], _dist[14]);
		_dist[6] = MIN(b._dist[6], _dist[6]);
		_dist[15] = MAX(b._dist[15], _dist[15]);
		_dist[7] = MIN(b._dist[7], _dist[7]);
		_dist[16] = MAX(b._dist[16], _dist[16]);
		_dist[8] = MIN(b._dist[8], _dist[8]);
		_dist[17] = MAX(b._dist[17], _dist[17]);
		return *this;
	}

	inline kDOP18 operator + (const kDOP18& v) const
	{
		kDOP18 rt(*this); return rt += v;
	}

	inline double length(int i) const {
		return _dist[i + 9] - _dist[i];
	}

	inline double width()  const { return _dist[9] - _dist[0]; }
	inline double height() const { return _dist[10] - _dist[1]; }
	inline double depth()  const { return _dist[11] - _dist[2]; }
	inline double volume() const { return width() * height() * depth(); }

	inline vec3f center() const {
		return { (_dist[0] + _dist[9]) * 0.5, (_dist[1] + _dist[10]) * 0.5, (_dist[2] + _dist[11]) * 0.5 };
	}

	inline double center(int i) const {
		return (_dist[i + 9] + _dist[i]) * 0.5f;
	}

	inline void empty() {
		for (int i = 0; i < 9; i++) {
			_dist[i] = FLT_MAX;
			_dist[i + 9] = -FLT_MAX;
		}
	}
};


kDOP18 vert_box(const Vert* vert, bool ccd);
kDOP18 node_box(const Node* node, bool ccd);
kDOP18 edge_box(const Edge* edge, bool ccd);
kDOP18 face_box(const Face* face, bool ccd);
kDOP18 dilate(const kDOP18& box, double d);
bool overlap(const kDOP18& box0, const kDOP18& box1, double thickness);


class DeformBVHNode {
public:
	kDOP18 _box;

	Face* _face;

	DeformBVHNode* _parent;
	DeformBVHNode* _left;
	DeformBVHNode* _right;

	bool _active;

public:
	DeformBVHNode();
	DeformBVHNode(DeformBVHNode*, Face*, kDOP18*, vec3f*);
	DeformBVHNode(DeformBVHNode*, Face**, unsigned int, kDOP18*, vec3f*);

	~DeformBVHNode();

	void refit(bool = false);
	bool find(Face*);

	inline DeformBVHNode* getLeftChild() { return _left; }
	inline DeformBVHNode* getRightChild() { return _right; }
	inline DeformBVHNode* getParent() { return _parent; }

	inline Face* getFace() { return _face; }
	inline bool isLeaf() { return _left == NULL; }
	inline bool isRoot() { return _parent == NULL; }

	friend class DeformBVHTree;
};

class DeformBVHTree {
public:
	Mesh* _mdl;
	DeformBVHNode* _root;
	Face** face_buffer;

	bool _ccd;

public:
	DeformBVHTree(Mesh&, bool);
	~DeformBVHTree();

	void Construct();

	double refit();

	//void do_task_1();
	//void do_task_2();

	kDOP18 box();
	inline DeformBVHNode* getRoot() { return _root; }

	friend class DeformBVHNode;
};