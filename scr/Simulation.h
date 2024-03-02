#pragma once

#include "Cloth.h"
#include "Obstacle.h"
#include "Handle.h"
#include "Collision.h"

struct Wind
{
	Tensor density;
	Tensor velocity;
	Tensor drag;
};

struct Simulation
{
public:

	bool dev_wordy{ true };

	bool is_noise_force{ false };

	int device_idx {0};

	torch::TensorOptions tensor_opt;
	
	int step, frame;

	Tensor step_time;

	Tensor step_time_cuda;

	Tensor falling_velocity;

	std::vector<Cloth> cloths;
	std::vector<Obstacle> obstacles;

	std::vector<Handle*> handles;
	std::vector<Constraint*> cons;

	Tensor gravity;

	Wind wind;

	Tensor noise_force;

	Simulation& operator=(const Simulation& sim)
	{
		std::cout << "Assignment Constructor" << std::endl;
		
		step = sim.step;
		frame = sim.frame;

		cloths = sim.cloths;
		obstacles = sim.obstacles;

		for (int h = 0; h < sim.handles.size(); ++h)
		{
			handles.push_back(sim.handles[h]);
		}

		for (int c = 0; c < sim.cons.size(); ++c)
		{
			cons.push_back(cons[c]);
		}

		step_time = sim.step_time;

		gravity = sim.gravity;
		wind = sim.wind;
		noise_force = sim.noise_force;
		
		return *this;
	}

	void clean_mesh() {
		// Delete Cloth Mesh
		for (int c_m = 0; c_m < cloths.size(); ++c_m) {
			delete_mesh(cloths[c_m].mesh);
		}

		for (int o_m = 0; o_m < obstacles.size(); ++o_m){
			delete_mesh(obstacles[o_m].mesh);	// Delete Obstacle Mesh
		}
	}

	void prepare();

	void prepare_cuda();

	void init_vectorization();

	void init_vectorization_cuda();

	void advance_step();

	std::pair<Tensor, Tensor> advance_step(const Tensor& pos, const Tensor& vel);

	void advance_step_cuda();

	std::pair<Tensor, Tensor> advance_step_cuda(const Tensor& pos, const Tensor& vel);

	void physics_step();

	void physics_step_cuda();

	void obstacle_step();

	void collision_step();

	void stable_step();

	void get_constraints();

	void delete_constraints();
};