#include "DDE.h"

#include <cmath>
#include <limits>
#include "Constants.h"

using namespace torch::indexing;

constexpr int nsamples = 10;

Tensor stretching_stiffness(const Tensor& G, const Tensor samples)
{
	Tensor a = G[0] + 0.25;
	Tensor b = G[3] + 0.25;
	Tensor c = torch::abs(G[1]);

	Tensor grid = torch::stack({ c, b, a }).reshape({ 1, 1, 1, 1, 3 }) * (nsamples * 2 / (nsamples - 1.0)) - 1;

	Tensor stiffness = torch::grid_sampler(samples, grid, 0, 1, true).squeeze();

	return stiffness;
}

Tensor evaluate_stretching_samples( const Tensor& stretch_ori, const Tensor& grid)
{
	if (stretch_ori.dim() == 2) {
		std::cout << "Homogeneous Stretch Sampling" << std::endl;
		Tensor data = torch::cat({ stretch_ori[0].view({ -1, 1, 1 }).repeat({ 1,1,5 }), stretch_ori.slice(0, 1, 6).t().unsqueeze(1) }, 1);
		Tensor stretch_tensor = torch::grid_sampler(data.unsqueeze(0), grid, 0, 1, true).reshape({ 1, 4, nsamples, nsamples, nsamples });
		return torch::relu(stretch_tensor * 2);
	}
	else if (stretch_ori.dim() == 3) {	
		// std::cout << "Heterogeneous Stretch Sampling" << std::endl;
		Tensor data = torch::cat({  
			stretch_ori.slice(1, 0, 1).permute({ 0, 2, 1 }).unsqueeze(3).repeat({1, 1, 1, 5}), 
			stretch_ori.slice(1, 1, 6).permute({ 0, 2, 1 }).unsqueeze(2) }, 2);
		Tensor stretch_tensor = torch::grid_sampler(data, grid.repeat({ data.size(0), 1, 1, 1 }), 0, 1, true).reshape({ data.size(0), 4, nsamples, nsamples, nsamples });
		
		return torch::relu(stretch_tensor * 2);
	}else{
		std::cout << "Inccorrect stretching sampling" << std::endl;
		exit(1);
	}
}

Tensor batch_stretching_stiffness(const Tensor& G, const Tensor& samples)
{
	Tensor a = (G[0] + 0.25);
	Tensor b = (G[3] + 0.25);
	Tensor c = torch::abs(G[1]);

	if (samples.size(0) == 1)
	{
		Tensor grid = torch::stack({ c, b, a }, 1).reshape({ 1, 1, 1, -1, 3 }) * (nsamples * 2 / (nsamples - 1.0)) - 1;
		return torch::grid_sampler(samples, grid, 0, 1, true).squeeze();
	}
	else if (samples.size(0) > 1)
	{
		Tensor grid_heter = torch::stack({ c, b, a }, 1).reshape({ -1, 1, 1, 1, 3 }) * (nsamples * 2 / (nsamples - 1.0)) - 1;
		return torch::grid_sampler(samples, grid_heter, 0, 1, true).squeeze().t();
	}
}

Tensor batch_stretching_stiffness_cuda(const Tensor& G, const Tensor& samples)
{
	Tensor a = (G[0] + 0.25);
	Tensor b = (G[3] + 0.25);
	Tensor c = torch::abs(G[1]);

	if (samples.size(0) == 1)
	{
		Tensor grid = torch::stack({ c, b, a }, 1).reshape({ 1, 1, 1, -1, 3 }) * (nsamples * 2 / (nsamples - 1.0)) - 1;
		return torch::grid_sampler(samples, grid, 0, 1, true).squeeze();
	}
	else if (samples.size(0) > 1)
	{
		Tensor grid_heter = torch::stack({ c, b, a }, 1).reshape({ -1, 1, 1, 1, 3 }) * (nsamples * 2 / (nsamples - 1.0)) - 1;
		return torch::grid_sampler(samples, grid_heter, 0, 1, true).squeeze().t();
	}
}

Tensor bending_stiffness(const Edge* edge, const Tensor& data0, const Tensor& data1)
{
	Tensor curv = edge->theta * edge->ldaa * 0.05;	// ?? Why multiply 0.05
	Tensor value = torch::clamp(curv - 1, -1, 1);
	// 0
	Tensor bias_angle0 = edge->bias_angle[0];
	Tensor grid0 = torch::stack({ value, bias_angle0 }).reshape({ 1, 1, 1, 2 });
	Tensor actual_ke0 = torch::relu(torch::grid_sampler(data0, grid0, 0, 2, true)).squeeze();
	// 1
	Tensor bias_angle1 = edge->bias_angle[1];
	if ((data0 == data1).all().item<bool>() && (bias_angle0 == bias_angle1).all().item<bool>())
		return actual_ke0;
	Tensor grid1 = torch::stack({ value, bias_angle1 }).reshape({ 1, 1, 1, 2 });
	Tensor actual_ke1 = torch::relu(torch::grid_sampler(data1, grid1, 0, 2, true)).squeeze();

	return torch::min(actual_ke0, actual_ke1);
}

Tensor batch_bending_stiffness(Tensor curv, Tensor bang, Tensor bend)
{
	Tensor value = torch::clamp(curv - 1, -1, 1); // because samples are per 0.05 cm^-1 = 5 m^-1

	if (bend.size(0) == 1)
	{
		Tensor grid = torch::stack({ value, bang }, 1).reshape({ 1, 1, -1, 2 }); 
		return torch::relu(grid_sampler(bend, grid, 0, 2, true).squeeze());
	}
	else if (bend.size(0) > 1)
	{
		Tensor grid_heter = torch::stack({ value, bang }, 1).reshape({ -1, 1, 1, 2 });
		return torch::relu(grid_sampler(bend, grid_heter, 0, 2, true).squeeze());
	}
}

Tensor batch_bending_stiffness(Tensor curv, Tensor bang, Tensor bend0, Tensor bend1)
{
	Tensor value = torch::clamp(curv - 1, -1, 1); // because samples are per 0.05 cm^-1 = 5 m^-1
	bend0 = bend0.squeeze(1);
	bend1 = bend1.squeeze(1);
	//0
	Tensor bias_angle0 = bang[0]; //(atan2(du0[1], du0[0]))*(4/M_PI);
	Tensor grid0 = torch::stack({ value, bias_angle0 }, 1).reshape({ -1,1,1,2 });
	Tensor actual_ke0 = torch::relu(grid_sampler(bend0, grid0, 0, 2, true).squeeze());
	//1
	Tensor bias_angle1 = bang[1]; //(atan2(du1[1], du1[0]))*(4/M_PI);
	if ((bias_angle0 == bias_angle1).all().item<bool>())
		return actual_ke0;
	Tensor grid1 = torch::stack({ value, bias_angle1 }, 1).reshape({ -1,1,1,2 });
	Tensor actual_ke1 = torch::relu(grid_sampler(bend1, grid1, 0, 2, true).squeeze());

	return torch::min(actual_ke0, actual_ke1);
}

Tensor batch_bending_stiffness_cuda(const Tensor& curv, const Tensor& bang, const Tensor& bend)
{
	Tensor value = torch::clamp(curv - 1, -1, 1); // because samples are per 0.05 cm^-1 = 5 m^-1

	if (bend.size(0) == 1)
	{
		Tensor grid = torch::stack({ value, bang }, 1).reshape({ 1, 1, -1, 2 });
		return torch::relu(grid_sampler(bend, grid, 0, 2, true).squeeze());
	}
	else if (bend.size(0) > 1)
	{
		Tensor grid_heter = torch::stack({ value, bang }, 1).reshape({ -1, 1, 1, 2 });
		return torch::relu(grid_sampler(bend, grid_heter, 0, 2, true).squeeze());
	}
}