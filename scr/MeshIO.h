#pragma once

#include "Mesh.h"

void read_obj(std::string filename, Mesh& mesh);

void save_obj(std::string filename, Mesh& mesh);