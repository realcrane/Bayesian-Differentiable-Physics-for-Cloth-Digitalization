#pragma once

#include "Mesh.h"

Tensor face_normal(const Face* face);

Tensor dihedral_angle(const Edge* edge);

Tensor unwrap_angle(Tensor theta, Tensor theta_ref);

Tensor signed_vf_distance(const Tensor& x, const Tensor& y0, const Tensor& y1, const Tensor& y2, Tensor* n, Tensor* w, double thres, bool& over);

Tensor signed_ee_distance(const Tensor& x0, const Tensor& x1, const Tensor& y0, const Tensor& y1, Tensor* n, Tensor* w, double thres, bool& over);

Tensor sub_signed_vf_distance(const Tensor& y0, const Tensor& y1, const Tensor& y2, Tensor* n, Tensor* w, double thres, bool& over);

Tensor sub_signed_ee_distance(const Tensor& x1mx0, const Tensor& y0mx0, const Tensor& y1mx0, const Tensor& y0mx1, const Tensor& y1mx1, const Tensor& y1my0, Tensor* n, Tensor* w, double thres, bool& over);
