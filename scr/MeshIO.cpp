#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>

#include "MeshIO.h"

static const std::array<std::string, 3> PROCESS_PRIMARY = { "v", "vt", "f" };

void read_obj(std::string filename, Mesh& mesh)
{
	std::ifstream obj_file{ filename };

	std::string line;

	while (std::getline(obj_file, line))
	{

		std::vector<std::string> tokens;

		auto position = line.find(" ");

		auto primary_type = line.substr(0, position);

		auto find_result = std::find(PROCESS_PRIMARY.begin(), PROCESS_PRIMARY.end(), primary_type);

		if (find_result != PROCESS_PRIMARY.end())
		{
			std::string token;

			if (primary_type == "v")
			{
				std::stringstream ss{ line };

				float Position[3];

				int token_cnt = 0;

				while (std::getline(ss, token, ' '))
				{
					token_cnt++;

					if (token_cnt > 1)
						Position[token_cnt - 2] = std::stof(token);
				}

				mesh.add(new Node(torch::from_blob(Position, { 3 }).clone().to(TNOPT), ZERO3));

			}
			else if (primary_type == "vt")
			{
				std::stringstream ss{ line };

				float UV[2];

				int token_cnt = 0;

				while (std::getline(ss, token, ' '))
				{
					token_cnt++;

					if (token_cnt > 1)
						UV[token_cnt - 2] = std::stof(token);

				}

				mesh.add(new Vert(torch::from_blob(UV, { 2 }).clone().to(TNOPT)));

			}
			else if (primary_type == "f")
			{

				std::vector<Node*> nodes;
				std::vector<Vert*> verts;

				std::stringstream ss{ line };

				int token_cnt = 0;

				while (std::getline(ss, token, ' '))
				{

					token_cnt++;

					if (token_cnt > 1)
					{
						int deliPos = token.find("/");	// Find the Position of Delimiter
						int token_len = token.length();	// The Length of the read token

						std::string nodeIdx = token.substr(0, deliPos);		// The index of "v" in the .obj file
						std::string vertIdx = token.substr(deliPos + 1, token_len);	// The index of "vt" in the .obj file

						int node_idx = std::stoi(nodeIdx) - 1;
						int vert_idx = std::stoi(vertIdx) - 1;

						nodes.push_back(mesh.nodes[node_idx]);
						verts.push_back(mesh.verts[vert_idx]);
					}
				}

				// Connect Verts and Nodes
				for (int v = 0; v < verts.size(); ++v)
				{
					// let the member node in vert points to the node
					verts[v]->node = nodes[v];
					// let the add the vert to the verts of node class
					auto find_vert = find(nodes[v]->verts.cbegin(), nodes[v]->verts.cend(), verts[v]);
					// add only if the vert does not exits in the node's verts vertor
					if (find_vert == nodes[v]->verts.cend())
						nodes[v]->verts.push_back(verts[v]);
				}

				mesh.add(new Face(verts[0], verts[1], verts[2]));

			}
		}
	}

	mesh.compute_ms_data();
	mesh.compute_ws_data();
}

void save_obj(std::string filename, Mesh& mesh)
{
	std::fstream file(filename, std::fstream::out);

	for (int v = 0; v < mesh.verts.size(); ++v)
	{
		const Vert* vert = mesh.verts[v];

		file << "vt " << vert->u[0].item<double>() << " " << vert->u[1].item<double>() << "\n";
	}

	for (int n = 0; n < mesh.nodes.size(); ++n)
	{
		const Node* node = mesh.nodes[n];

		file << "v " << node->x[0].item<double>() << " "
			<< node->x[1].item<double>() << " "
			<< node->x[2].item<double>() << "\n";
	}

	for (int f = 0; f < mesh.faces.size(); ++f)
	{
		const Face* face = mesh.faces[f];

		file << "f " << face->v[0]->node->index + 1 << "/" << face->v[0]->index + 1
			<< " " << face->v[1]->node->index + 1 << "/" << face->v[1]->index + 1
			<< " " << face->v[2]->node->index + 1 << "/" << face->v[2]->index + 1 << "\n";
	}

	file.close();
}