// geometry_ops.h
//
// C++ re-implementation of diffusion_net.geometry.compute_operators()
// (using potpourri3d's cotan_laplacian / vertex_areas + scipy eigsh),
// so that a C++ deployment can compute the geometric operators directly from
// a mesh (verts/faces) without Python. All formulas match the Python version.
//
// Inputs:  verts (V,3) float64, faces (F,3) int32/64 (triangular).
// Outputs: massvec (V,), L (VxV sparse, CSR), evals (K,), evecs (V,K),
//          gradX/gradY (VxV sparse)  -- same as get_operators().
//
#include <vector>
#include <array>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <cstdint>

// Vertex areas (lumped mass, barycentric): vertex_area = sum(adjacent face areas)/3
std::vector<double> vertex_areas(const std::vector<std::array<double,3>>& V,
                                 const std::vector<std::array<int32_t,3>>& F);

// Cotan Laplacian (weak, CSR). Returns (V x V) sparse matrix in column-major.
Eigen::SparseMatrix<double> cotan_laplacian(const std::vector<std::array<double,3>>& V,
                                            const std::vector<std::array<int32_t,3>>& F,
                                            double denom_eps = 1e-10);

// Generalized eigendecomposition: L v = lambda M v, first k eigenpairs near sigma.
// M is a diagonal matrix given by its diagonal vector massvec.
// Returns evals (k,) and evecs (V,k) with evecs as columns of a (V x k) matrix.
// On success returns true. Matches scipy.sparse.linalg.eigsh(L+k*eps*I, k=k, M=diag(massvec), sigma=eps).
bool generalized_eigs(const Eigen::SparseMatrix<double>& L,
                      const std::vector<double>& massvec,
                      int k, double eps, double sigma,
                      std::vector<double>& evals,
                      Eigen::MatrixXd& evecs);

// Build per-vertex tangent frames: frames[i] is (3x3) with rows = X/Y/Z basis,
// Z = vertex normal, X/Y orthonormal tangent. Same as build_tangent_frames().
std::vector<std::array<double,9>> build_tangent_frames(
    const std::vector<std::array<double,3>>& V,
    const std::vector<std::array<int32_t,3>>& F);

// Compute gradient operator (complex, stored as separate real/imag sparse mats).
// Returns gradX (real), gradY (imag) as (V x V) sparse matrices.
// Matches build_grad() + edge_tangent_vectors().
void build_grad_ops(const std::vector<std::array<double,3>>& V,
                    const std::vector<std::array<int32_t,3>>& F,
                    Eigen::SparseMatrix<double>& gradX,
                    Eigen::SparseMatrix<double>& gradY);
// Normalize vertex positions in place: center by mean, then unit-scale by max radius.
// Matches diffusion_net.geometry.normalize_positions(method='mean', scale_method='max_rad').
void normalize_positions(std::vector<std::array<double,3>>& V);

// Extract COO triplets (rows/cols int64, vals double) from a sparse matrix.
// Matches scipy coo_matrix(M).row/.col/.data -- feeds the ONNX sparse-MM gather inputs.
void sparse_to_coo(const Eigen::SparseMatrix<double>& M,
                   std::vector<int64_t>& rows,
                   std::vector<int64_t>& cols,
                   std::vector<double>& vals);
