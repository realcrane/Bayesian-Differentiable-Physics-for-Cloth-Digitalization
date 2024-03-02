#include "Geometry.h"

Tensor norm2(Tensor x)
{
	return torch::dot(x, x);
}

Tensor normalize(Tensor x)
{
	Tensor l = torch::norm(x);

	if (l.item<double>() == 0.0)
	{
		std::cout << "Zero Length in Normalization" << std::endl;
		return x;
	}

	return x / l;
}


Tensor stp(Tensor u, Tensor v, Tensor w)
{
	return torch::dot(u, torch::cross(v, w));
}

Tensor face_normal(const Face* face)
{
	const Tensor x0 = face->v[0]->node->x;
	const Tensor x1 = face->v[1]->node->x;
	const Tensor x2 = face->v[2]->node->x;

	Tensor face_n = normalize(torch::cross(x1 - x0, x2 - x0));

	return face_n;
}

Tensor unwrap_angle(Tensor theta, Tensor theta_ref)
{
	if (((theta - theta_ref) > M_PI).item<bool>())
	{
		theta -= 2 * M_PI;
	}
	if (((theta - theta_ref) < -M_PI).item<bool>())
	{
		theta += 2 * M_PI;
	}
	return theta;
}

Tensor dihedral_angle(const Edge* edge)
{
	//std::cout << "Compute Dihedral Angle" << std::endl;
	
	if (edge->adj_faces[0] == nullptr || edge->adj_faces[1] == nullptr)
		return ZERO;

	Tensor edge_length_ws = edge->nodes[0]->x - edge->nodes[1]->x;
	Tensor edge_lenght_ws_l = torch::norm(edge_length_ws);
	Tensor edge_length_ws_normal = edge_length_ws;

	if (edge_lenght_ws_l.item<double>() != 0.0)
		edge_length_ws_normal = edge_length_ws / edge_lenght_ws_l;

	if ((edge_length_ws_normal == 0.0).all().item<bool>())
		return ZERO;

	Tensor face0_normal = face_normal(edge->adj_faces[0]);
	Tensor face1_normal = face_normal(edge->adj_faces[1]);

	if ((face0_normal == 0.0).all().item<bool>() || (face1_normal == 0.0).all().item<bool>())
		return ZERO;

	Tensor cosine = torch::dot(face0_normal, face1_normal);
	Tensor sine = torch::dot(edge_length_ws_normal, torch::cross(face0_normal, face1_normal));

	Tensor theta = torch::atan2(sine, cosine);

	return unwrap_angle(theta, edge->reference_angle);	
}

Tensor signed_vf_distance(const Tensor& x, const Tensor& y0, const Tensor& y1, const Tensor& y2, Tensor* n, Tensor* w, double thres, bool& over)
{
	return sub_signed_vf_distance(y0 - x, y1 - x, y2 - x, n, w, thres, over);
}

Tensor signed_ee_distance(const Tensor& x0, const Tensor& x1,const Tensor& y0, const Tensor& y1, Tensor* n, Tensor* w, double thres, bool& over) 
{
	return sub_signed_ee_distance(x1 - x0, y0 - x0, y1 - x0, y0 - x1, y1 - x1, y1 - y0, n, w, thres, over);
}

Tensor sub_signed_vf_distance(const Tensor& y0, const Tensor& y1, const Tensor& y2, Tensor* n, Tensor* w, double thres, bool& over)
{
	Tensor _n; 

	if (!n) {
		n = &_n;
	}

	Tensor _w[4]; 
	if (!w)
	{
		w = _w;
	}

	*n = torch::cross(y1 - y0, y2 - y0);

	w[0] = w[1] = w[2] = w[3] = ZERO;

	if (norm2(*n).item<double>() < 1e-6) 
	{
		over = true;
		return INFINITY_TENSOR;
	}

	*n = normalize(*n);

	Tensor h = -torch::dot(y0, *n);

	over = torch::abs(h).item<double>() > thres;

	if (over)
	{
		return h;
	}

	Tensor b0 = stp(y1, y2, *n);
	Tensor b1 = stp(y2, y0, *n);
	Tensor b2 = stp(y0, y1, *n);

	Tensor sum = 1 / (b0 + b1 + b2);

	w[0] = ONE;
	w[1] = -b0 * sum;
	w[2] = -b1 * sum;
	w[3] = -b2 * sum;

	return h;
}

Tensor sub_signed_ee_distance(const Tensor& x1mx0, const Tensor& y0mx0, const Tensor& y1mx0, const Tensor& y0mx1, const Tensor& y1mx1, const Tensor& y1my0, Tensor* n, Tensor* w, double thres, bool& over)
{
	Tensor _n; 
	
	if (!n)
	{
		n = &_n;
	}

	Tensor _w[4]; 
	
	if (!w)
	{
		w = _w;
	}

	*n = torch::cross(x1mx0, y1my0);

	w[0] = w[1] = w[2] = w[3] = ZERO;

	if ((norm2(*n).item<double>() < 1e-6)) 
	{
		over = true;
		return INFINITY_TENSOR;
	}

	*n = normalize(*n);
	Tensor h = -torch::dot(y0mx0, *n);

	over = abs(h).item<double>() > thres;

	if (over)
	{
		return h;
	}

	Tensor a0 = stp(y1mx1, y0mx1, *n);		
	Tensor a1 = stp(y0mx0, y1mx0, *n);
	Tensor b0 = stp(y1mx1, y1mx0, *n);
	Tensor b1 = stp(y0mx0, y0mx1, *n);

	Tensor suma = 1 / (a0 + a1);
	Tensor sumb = 1 / (b0 + b1);

	w[0] = a0 * suma;
	w[1] = a1 * suma;
	w[2] = -b0 * sumb;
	w[3] = -b1 * sumb;

	return h;
}