#pragma once
#include "torch/extension.h"

#include "Constants.h"

using torch::Tensor;

inline size_t find_index(int i, const std::vector<int>& indices)
{
	for (size_t ii = 0; ii < indices.size(); ++ii)
	{
		if (indices[ii] == i)
			return ii;
	}
	return indices.size();
}

inline void insert_index(int i, int j, std::vector<int>& indices, std::vector<Tensor>& entries) 
{
	indices.insert(indices.begin() + j, i);
	entries.insert(entries.begin() + j, ZERO33.clone());
}

struct SparseVector 
{
	std::vector<int> indices;
	std::vector<Tensor> entries;

	Tensor operator[](int i) const
	{
		size_t j = find_index(i, indices);
		if (j >= indices.size() || indices[j] != j)
			return ZERO33;
		else
			return entries[j];
	}

	Tensor& operator[] (int i)
	{
		// inserts entry as side-effect

		size_t j = find_index(i, indices);
		if (j >= indices.size() || indices[j] != i)
			insert_index(static_cast<int>(i), static_cast<int>(j), indices, entries);
		return entries[j];
	}

};

struct SparseMatrix
{
	int m, n;
	std::vector<SparseVector> rows;
	
	SparseMatrix(int _m, int _n):
		m{ _m }, n{ _n }
	{
		rows = std::vector<SparseVector>(_m);
	}

	Tensor operator() (int i, int j) const
	{
		return rows[i][j];
	}

	Tensor& operator() (int i, int j)
	{
		// inserts entry as side-effect
		return rows[i][j];
	}
};

void add_submat(SparseMatrix& A, int i, int j, const Tensor& Aij);

void add_submat(const Tensor& Asub, const std::vector<int>& ix, SparseMatrix& A);

void add_submat(const Tensor& Asub, const std::vector<int>& ix, Tensor& A);

void add_submat_unfold(const Tensor& Asub_unfold, const std::vector<int>& ix, SparseMatrix& A);

void add_subvec(const Tensor& bsub, const std::vector<int>& ix, Tensor& b);

void add_subvec(const Tensor& bsub, const std::vector<int>& ix, std::vector<Tensor>& b);

void add_subvec_unfold(const Tensor& bsub_unfold, const std::vector<int>& ix, Tensor& b);

std::ostream& operator<< (std::ostream& out, const SparseMatrix& A);

Tensor EigenSolverForward(const Tensor& A, const Tensor& b, bool verb = false);

std::vector<Tensor> EigenSolverBackward(Tensor& dldz, const Tensor& ans, const Tensor& tensor_A, const Tensor& tensor_b);

Tensor EigenSolverForwardCUDA(const Tensor& A, const Tensor& b, bool verb = false);

std::vector<Tensor> EigenSolverBackwardCUDA(Tensor& dldz, const Tensor& ans, const Tensor& tensor_A, const Tensor& tensor_b);

class EigenSolver : public torch::autograd::Function<EigenSolver> {
public:

	static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor A, Tensor b) {
		auto output = EigenSolverForward(A, b);
		ctx->save_for_backward({ output, A, b });
		return output;
	}

	static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list dldz) {
		auto saved = ctx->get_saved_variables();
		auto ans = saved[0];
		auto A = saved[1];
		auto b = saved[2];
		std::vector<Tensor> ans_back = EigenSolverBackward(dldz[0], ans, A, b);
		return { ans_back[0], ans_back[1] };
	}
};

class EigenSolverCUDA : public torch::autograd::Function<EigenSolverCUDA> {
public:

	static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor A, Tensor b, int device_idx) {
		auto A_sparse_cpu = A.to_sparse().cpu();
		auto b_cpu = b.cpu();
		auto output = EigenSolverForwardCUDA(A_sparse_cpu, b_cpu);
		ctx->save_for_backward({ output, A_sparse_cpu, b_cpu });
		ctx->saved_data["device_idx"] = device_idx;
		return output;
	}

	static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list dldz) {
		auto saved = ctx->get_saved_variables();
		int device_idx = ctx->saved_data["device_idx"].toInt();
		auto ans = saved[0];
		auto A = saved[1];
		auto b = saved[2];
		std::vector<Tensor> ans_back = EigenSolverBackwardCUDA(dldz[0], ans, A, b);	
		at::Tensor undef;	// For None gradient tensor
		return { ans_back[0].to(torch::Device(torch::kCUDA, device_idx)), ans_back[1].to(torch::Device(torch::kCUDA, device_idx)), undef};
	}
};

Tensor pytorch_solve(const Tensor& Jaco, const Tensor& M, const Tensor& b, const Tensor& vs, const Tensor& dt, const int& num_nodes);

Tensor solve_cubic_forward(Tensor a, Tensor b, Tensor c, Tensor d);

std::vector<Tensor> solve_cubic_backward(Tensor dldz, Tensor ans, Tensor a, Tensor b, Tensor c, Tensor d);

class CubicSolver : public torch::autograd::Function<CubicSolver> {
public:

	static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor a, Tensor b, Tensor c, Tensor d) {
		auto ans = solve_cubic_forward(a, b, c, d);
		ctx->save_for_backward({ ans, a, b, c, d });
		return ans;
	}

	static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list dldz) {
		auto saved = ctx->get_saved_variables();
		auto ans = saved[0];
		auto a = saved[1];
		auto b = saved[2];
		auto c = saved[3];
		auto d = saved[4];
		std::vector<Tensor> ans_back = solve_cubic_backward(dldz[0], ans, a, b, c, d);
		return { ans_back[0], ans_back[1], ans_back[2], ans_back[3] };
	}
};
