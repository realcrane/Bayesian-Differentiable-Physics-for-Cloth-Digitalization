#include "Obstacle.h"

void Obstacle::compute_masses()
{
	for (int v = 0; v < mesh.verts.size(); ++v)
		mesh.verts[v]->m = mass;

	for (int n = 0; n < mesh.nodes.size(); ++n)
		mesh.nodes[n]->m = mass;

}