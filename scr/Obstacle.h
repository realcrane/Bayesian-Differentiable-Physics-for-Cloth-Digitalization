#pragma once
#include "Mesh.h"

struct Obstacle 
{
	Mesh mesh;

	Tensor mass;

	void compute_masses();
};