#include "Eigen/Eigen"
#include "Eigen/SparseCore"
// #include "Eigen/CholmodSupport"

#include "Solve.h"

#include <chrono>

using namespace std::chrono;

void add_submat(SparseMatrix& A, int i, int j, const Tensor& Aij)
{
	A(i, j) = A(i, j) + Aij;
}

void add_submat(const Tensor& Asub, const std::vector<int>& ix, SparseMatrix& A)
{
	int m = ix.size();

	for (int i = 0; i < m; ++i)
	{
		const Tensor& tmp = Asub.slice(0, i * 3, i * 3 + 3);

		for (int j = 0; j < m; ++j)
		{
			A(ix[i], ix[j]) = A(ix[i], ix[j]) + tmp.slice(1, j * 3, j * 3 + 3);
		}
	}
}

void add_submat(const Tensor& Asub, const std::vector<int>& ix, Tensor& A)
{
	int m = ix.size();

	for (int i = 0; i < m; ++i)
	{
		const Tensor& tmp = Asub.slice(0, i * 3, i * 3 + 3);

		for (int j = 0; j < m; ++j)
		{
			A[i][j] = A[i][j] + tmp.slice(1, j * 3, j * 3 + 3);
		}
	}
}

void add_submat_unfold(const Tensor& Asub_unfold, const std::vector<int>& ix, SparseMatrix& A)
{
	int m = ix.size();

	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < m; ++j) {			
			A(ix[i], ix[j]) = A(ix[i], ix[j]) + Asub_unfold[i][j];
		}
	}
}


void add_subvec(const Tensor& bsub, const std::vector<int>& ix, Tensor& b)
{
	int m = ix.size();

	for (int i = 0; i < m; ++i)
	{
		b[ix[i]] = b[ix[i]] + bsub.slice(0, i * 3, i * 3 + 3);
	}
}

void add_subvec(const Tensor& bsub, const std::vector<int>& ix, std::vector<Tensor>& b)
{
	int m = ix.size();

	for (int i = 0; i < m; ++i)
	{
		b[ix[i]] = b[ix[i]] + bsub.slice(0, i * 3, i * 3 + 3);
	}
}

void add_subvec_unfold(const Tensor& bsub_unfold, const std::vector<int>& ix, Tensor& b)
{
	int m = ix.size();

	for (int i = 0; i < m; ++i)
	{		
		b[ix[i]] = b[ix[i]] + bsub_unfold[i];
	}
}

std::ostream& operator<< (std::ostream& out, const SparseMatrix& A) 
{
	out << "[";

	for (int i = 0; i < A.m; i++) 
	{
		
		const SparseVector& row = A.rows[i];

		for (int jj = 0; jj < row.indices.size(); jj++) 
		{
			int j = row.indices[jj];
			const Tensor& aij = row.entries[jj];
			out << (i == 0 && jj == 0 ? "" : ", ") << "(" << i << "," << j
				<< "): " << aij;
		}
	}
	out << "]";

	return out;
}


Tensor pytorch_solve(const Tensor& Jaco, const Tensor& M, const Tensor& b, const Tensor& vs, const Tensor& dt, const int& num_nodes)
{

	auto A = M - dt * dt * Jaco;
	auto B = dt * (b + dt * torch::matmul(Jaco, vs));

	auto x = EigenSolver::apply(A, B);

	x = x.reshape({ num_nodes, 3 });

	return x;

	//return EigenSolver::apply(M - dt * dt * Jaco, dt * (b + dt * torch::matmul(Jaco, vs))).reshape({ num_nodes, 3 });
}

Tensor EigenSolverForward(const Tensor& A, const Tensor& b, bool verb) {

	
	// Tensor A to Eigen Sparse Matrix SpaA

	Tensor sparse = A.to_sparse();

	Tensor flat_idx = sparse.indices().flatten();
	Tensor sparse_values = sparse.values();

	std::vector<int64_t> row_idxs(flat_idx.data_ptr<int64_t>(), flat_idx.data_ptr<int64_t>() + flat_idx.numel() / 2);
	std::vector<int64_t> col_idxs(flat_idx.data_ptr<int64_t>() + flat_idx.numel() / 2, flat_idx.data_ptr<int64_t>() + flat_idx.numel());
	std::vector<double> values(sparse_values.data_ptr<double>(), sparse_values.data_ptr<double>() + sparse_values.numel());

	std::vector<Eigen::Triplet<double>> trips;

	for (int i = 0; i < values.size(); ++i) {
		trips.emplace_back(row_idxs[i], col_idxs[i], values[i]);
	}

	Eigen::SparseMatrix<double> SpaA(A.size(0), A.size(1));
	SpaA.setFromTriplets(trips.begin(), trips.end());

	// Tensor vector b to Eigen vector Spab
	const double* b_data = b.data_ptr<double>();
	Eigen::Map<const Eigen::VectorXd> Spab(b_data, b.size(0));
	// Tensor vector new velocity to Eigen vector x
	Tensor new_vel{ torch::zeros_like(b, TNOPT) };
	double* x_data = new_vel.data_ptr<double>();
	Eigen::Map<Eigen::VectorXd> x(x_data, new_vel.size(0));

	// Eigen CG solve  
	//Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
	//cg.compute(SpaA);
	//x = cg.solve(Spab);

	// Eigen LLT solver
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper> ldlt;
	ldlt.compute(SpaA);
	x = ldlt.solve(Spab);

	//Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> chollt;
	//chollt.compute(SpaA);
	//x = chollt.solve(Spab);

	//Eigen::CholmodSimplicialLDLT<Eigen::SparseMatrix<double>> chollt;
	//chollt.compute(SpaA);
	//x = chollt.solve(Spab);

	//end = high_resolution_clock::now();
	//std::cout << "Eigen Solve time: " << duration_cast<milliseconds>(end - start).count() << std::endl;

	//if (verb) {
		//std::cout << "#iterations:     " << cg.iterations() << std::endl;
		//std::cout << "estimated error: " << cg.error() << std::endl;
		//std::cout << x << std::endl;
	//}

	return new_vel;
}

std::vector<Tensor> EigenSolverBackward(Tensor& dldz, const Tensor& ans, const Tensor& tensor_A, const Tensor& tensor_b) {

	//auto start = high_resolution_clock::now();

	Tensor sparse = tensor_A.to_sparse();

	Tensor flat_idx = sparse.indices().flatten();
	Tensor sparse_values = sparse.values();

	std::vector<int64_t> row_idxs(flat_idx.data_ptr<int64_t>(), flat_idx.data_ptr<int64_t>() + flat_idx.numel() / 2);
	std::vector<int64_t> col_idxs(flat_idx.data_ptr<int64_t>() + flat_idx.numel() / 2, flat_idx.data_ptr<int64_t>() + flat_idx.numel());
	std::vector<double> values(sparse_values.data_ptr<double>(), sparse_values.data_ptr<double>() + sparse_values.numel());

	std::vector<Eigen::Triplet<double>> trips;

	for (int i = 0; i < values.size(); ++i) {
		trips.emplace_back(row_idxs[i], col_idxs[i], values[i]);
	}

	Eigen::SparseMatrix<double> SpaA(tensor_A.size(0), tensor_A.size(1));
	SpaA.setFromTriplets(trips.begin(), trips.end());
	
	// Tensor tensor_A to Eigen Sparse Matrix SpaA
	//Eigen::SparseMatrix<double> SpaA(tensor_A.size(0), tensor_A.size(1));
	//SpaA.reserve(Eigen::VectorXi::Constant(tensor_A.size(1), 16));
	//auto A_acc = tensor_A.accessor<double, 2>();
	//for (int i = 0; i < tensor_A.size(0); ++i)
	//	for (int j = 0; j < tensor_A.size(1); ++j)
	//		if (A_acc[i][j] != 0.0) {
	//			SpaA.insert(i, j) = A_acc[i][j];
	//		}
	// Tensor vector tensor_b to Eigen vector Spab
	const double* b_data = dldz.contiguous().data_ptr<double>();
	Eigen::Map<const Eigen::VectorXd> Spab(b_data, tensor_b.size(0));
	// Tensor vector dx to Eigen vector x (recieve result)
	Tensor dx{ torch::zeros_like(tensor_b, TNOPT) };
	double* x_data = dx.data_ptr<double>();
	Eigen::Map<Eigen::VectorXd> x(x_data, dx.size(0));

	// Eigen CG solver
	//Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
	//cg.compute(SpaA);
	//x = cg.solve(Spab);

	// Eigen LLT solver
	Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper> llt;
	llt.compute(SpaA);
	x = llt.solve(Spab);

	Tensor dlda{ -torch::ger(dx.squeeze(), ans.squeeze()) };

	//auto end = high_resolution_clock::now();
	//std::cout << "Backward Eigen Solve time: " << duration_cast<milliseconds>(end - start).count() << std::endl;

	return { dlda, dx };
}

Tensor EigenSolverForwardCUDA(const Tensor& sparse, const Tensor& b, bool verb)
{
	// Tensor A to Eigen Sparse Matrix SpaA
	Tensor flat_idx = sparse.indices().flatten();
	Tensor sparse_values = sparse.values();

	std::vector<int64_t> row_idxs(flat_idx.data_ptr<int64_t>(), flat_idx.data_ptr<int64_t>() + flat_idx.numel() / 2);
	std::vector<int64_t> col_idxs(flat_idx.data_ptr<int64_t>() + flat_idx.numel() / 2, flat_idx.data_ptr<int64_t>() + flat_idx.numel());
	std::vector<double> values(sparse_values.data_ptr<double>(), sparse_values.data_ptr<double>() + sparse_values.numel());

	std::vector<Eigen::Triplet<double>> trips;

	for (int i = 0; i < values.size(); ++i) {
		trips.emplace_back(row_idxs[i], col_idxs[i], values[i]);
	}

	Eigen::SparseMatrix<double> SpaA(sparse.size(0), sparse.size(1));
	SpaA.setFromTriplets(trips.begin(), trips.end());

	// Tensor vector b to Eigen vector Spab
	const double* b_data = b.data_ptr<double>();
	Eigen::Map<const Eigen::VectorXd> Spab(b_data, b.size(0));

	// Tensor vector new velocity to Eigen vector x
	Tensor new_vel{ torch::zeros_like(b, TNOPT) };
	double* x_data = new_vel.data_ptr<double>();
	Eigen::Map<Eigen::VectorXd> x(x_data, new_vel.size(0));

	// Eigen LLT solver
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper> ldlt;
	ldlt.compute(SpaA);
	x = ldlt.solve(Spab);

	return new_vel;
}

std::vector<Tensor> EigenSolverBackwardCUDA(Tensor& dldz, const Tensor& ans, const Tensor& sparse, const Tensor& tensor_b)
{
	Tensor flat_idx = sparse.indices().flatten();
	Tensor sparse_values = sparse.values();

	std::vector<int64_t> row_idxs(flat_idx.data_ptr<int64_t>(), flat_idx.data_ptr<int64_t>() + flat_idx.numel() / 2);
	std::vector<int64_t> col_idxs(flat_idx.data_ptr<int64_t>() + flat_idx.numel() / 2, flat_idx.data_ptr<int64_t>() + flat_idx.numel());
	std::vector<double> values(sparse_values.data_ptr<double>(), sparse_values.data_ptr<double>() + sparse_values.numel());

	std::vector<Eigen::Triplet<double>> trips;

	for (int i = 0; i < values.size(); ++i) {
		trips.emplace_back(row_idxs[i], col_idxs[i], values[i]);
	}

	Eigen::SparseMatrix<double> SpaA(sparse.size(0), sparse.size(1));
	SpaA.setFromTriplets(trips.begin(), trips.end());

	// Tensor vector tensor_b to Eigen vector Spab
	const double* b_data = dldz.contiguous().data_ptr<double>();
	Eigen::Map<const Eigen::VectorXd> Spab(b_data, tensor_b.size(0));
	// Tensor vector dx to Eigen vector x (recieve result)
	Tensor dx{ torch::zeros_like(tensor_b, TNOPT) };
	double* x_data = dx.data_ptr<double>();
	Eigen::Map<Eigen::VectorXd> x(x_data, dx.size(0));

	// Eigen LLT solver
	Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper> llt;
	llt.compute(SpaA);
	x = llt.solve(Spab);

	Tensor dlda{ -torch::ger(dx.squeeze(), ans.squeeze()) };

	return { dlda, dx };
}

int solve_quadratic(double a, double b, double c, double x[2]) {
	
	double d = b * b - 4 * a * c;
	if (d < 0) {
		x[0] = -b / (2 * a);
		return 0;
	}
	double q = -(b + sqrt(d)) / 2;
	double q1 = -(b - sqrt(d)) / 2;
	int i = 0;
	if (abs(a) > 1e-12) {
		x[i++] = q / a;
		x[i++] = q1 / a;
	}
	else {
		x[i++] = -c / b;
	}
	if (i == 2 && x[0] > x[1])
		std::swap(x[0], x[1]);
	return i;
}

double newtons_method(double a, double b, double c, double d, double x0,
	int init_dir) {
	if (init_dir != 0) {
		// quadratic approximation around x0, assuming y' = 0
		double y0 = d + x0 * (c + x0 * (b + x0 * a)),
			ddy0 = 2 * b + (x0 + init_dir * 1e-6) * (6 * a);
		x0 += init_dir * sqrt(abs(2 * y0 / ddy0));
	}
	for (int iter = 0; iter < 100; iter++) {
		double y = d + x0 * (c + x0 * (b + x0 * a));
		double dy = c + x0 * (2 * b + x0 * 3 * a);
		if (dy == 0)
			return x0;
		double x1 = x0 - y / dy;
		if (abs(x0 - x1) < 1e-6)
			return x0;
		x0 = x1;
	}
	return x0;
}

Tensor solve_cubic_forward(Tensor aa, Tensor bb, Tensor cc, Tensor dd) {
	double a = aa.item<double>(), b = bb.item<double>();
	double c = cc.item<double>(), d = dd.item<double>();
	double xc[2], x[3];
	x[0] = x[1] = x[2] = -1;
	xc[0] = xc[1] = -1;
	int ncrit = solve_quadratic(3 * a, 2 * b, c, xc);
	if (ncrit == 0) {
		x[0] = newtons_method(a, b, c, d, xc[0], 0);
		return torch::tensor({ x[0] }, TNOPT);
	}
	else if (ncrit == 1) {// cubic is actually quadratic
		int nsol = solve_quadratic(b, c, d, x);
		return torch::tensor(std::vector<double>(x, x + nsol), TNOPT);
	}
	else {
		double yc[2] = { d + xc[0] * (c + xc[0] * (b + xc[0] * a)),
						d + xc[1] * (c + xc[1] * (b + xc[1] * a)) };
		int i = 0;
		if (yc[0] * a >= 0)
			x[i++] = newtons_method(a, b, c, d, xc[0], -1);
		if (yc[0] * yc[1] <= 0) {
			int closer = abs(yc[0]) < abs(yc[1]) ? 0 : 1;
			x[i++] = newtons_method(a, b, c, d, xc[closer], closer == 0 ? 1 : -1);
		}
		if (yc[1] * a <= 0)
			x[i++] = newtons_method(a, b, c, d, xc[1], 1);
		return torch::tensor(std::vector<double>(x, x + i), TNOPT);
	}
}

std::vector<Tensor> solve_cubic_backward(Tensor dldz, Tensor ans, Tensor a, Tensor b, Tensor c, Tensor d)
{
	Tensor dldd = dldz / (ans * (3 * a * ans + 2 * b) + c);
	Tensor dldc = dldd * ans;
	Tensor dldb = dldc * ans;
	Tensor dlda = dldb * ans;
	return { dlda, dldb, dldc, dldd };
}