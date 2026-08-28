// geometry_ops.cpp
#include "geometryOps.h"
#include <array>
#include <complex>
#include <cmath>
#include <vector>
#include <algorithm>
#include <Eigen/Eigenvalues>
#include <Spectra/SymGEigsShiftSolver.h>
#include <Spectra/MatOp/SymShiftInvert.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

using namespace Eigen;
using namespace Spectra;

static double dot3(const double* a, const double* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
static void cross3(const double* a, const double* b, double* out) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

// --- face areas: 0.5 *; (V[f1]-V[f0]) x (V[f2]-V[f0]); ---
std::vector<double> face_areas(const std::vector<std::array<double,3>>& V,
                               const std::vector<std::array<int32_t,3>>& F) {
    std::vector<double> areas(F.size(), 0.0);
    for (size_t f = 0; f < F.size(); ++f) {
        const double* a = V[F[f][0]].data();
        const double* b = V[F[f][1]].data();
        const double* c = V[F[f][2]].data();
        double e1[3] = {b[0]-a[0], b[1]-a[1], b[2]-a[2]};
        double e2[3] = {c[0]-a[0], c[1]-a[1], c[2]-a[2]};
        double cr[3]; cross3(e1, e2, cr);
        areas[f] = 0.5 * std::sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);
    }
    return areas;
}

// --- vertex areas (barycentric lumped mass) ---
std::vector<double> vertex_areas(const std::vector<std::array<double,3>>& V,
                                 const std::vector<std::array<int32_t,3>>& F) {
    std::vector<double> fa = face_areas(V, F);
    std::vector<double> va(V.size(), 0.0);
    for (size_t f = 0; f < F.size(); ++f)
        for (int i = 0; i < 3; ++i)
            va[F[f][i]] += fa[f];
    for (auto& x : va) x /= 3.0;
    return va;
}

// --- cotan Laplacian (weak), matches potpourri3d.cotan_laplacian(V,F,denom_eps) ---
Eigen::SparseMatrix<double> cotan_laplacian(const std::vector<std::array<double,3>>& V,
                                            const std::vector<std::array<int32_t,3>>& F,
                                            double denom_eps) {
    const int nV = (int)V.size();
    std::vector<Triplet<double>> trips;
    trips.reserve(F.size() * 12);
    for (size_t f = 0; f < F.size(); ++f) {
        for (int i = 0; i < 3; ++i) {
            int ii = F[f][i];
            int jj = F[f][(i+1)%3];
            int kk = F[f][(i+2)%3];
            const double* ki = V[ii].data();
            const double* kj = V[jj].data();
            const double* k = V[kk].data();
            double vki[3] = {ki[0]-k[0], ki[1]-k[1], ki[2]-k[2]};
            double vkj[3] = {kj[0]-k[0], kj[1]-k[1], kj[2]-k[2]};
            double d = dot3(vki, vkj);
            double cr[3]; cross3(vki, vkj, cr);
            double cm = std::sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);
            double w = 0.5 * d / (cm + denom_eps);
            // (i,i)+=w, (j,j)+=w, (i,j)-=w, (j,i)-=w
            trips.push_back({ii, ii, w});
            trips.push_back({jj, jj, w});
            trips.push_back({ii, jj, -w});
            trips.push_back({jj, ii, -w});
        }
    }
    Eigen::SparseMatrix<double> L(nV, nV);
    L.setFromTriplets(trips.begin(), trips.end());
    return L;
}
// --- generalized eigendecomposition via Spectra (matches scipy eigsh) ---
bool generalized_eigs(const Eigen::SparseMatrix<double>& L,
                      const std::vector<double>& massvec,
                      int k, double eps, double sigma,
                      std::vector<double>& evals,
                      Eigen::MatrixXd& evecs) {
    const int n = L.rows();
    // Build (L + eps*I) and M = diag(massvec)
    Eigen::SparseMatrix<double> A = L;
    for (int i = 0; i < n; ++i) A.coeffRef(i, i) += eps;
    Eigen::SparseMatrix<double> M(n, n);
    std::vector<Triplet<double>> tm;
    tm.reserve(n);
    for (int i = 0; i < n; ++i) tm.push_back({i, i, massvec[i]});
    M.setFromTriplets(tm.begin(), tm.end());

    using OpType = SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
    using BOpType = SparseSymMatProd<double>;
    OpType op(A, M);
    BOpType Bop(M);

    SymGEigsShiftSolver<OpType, BOpType, GEigsMode::ShiftInvert> geigs(op, Bop, k, 2*k+10, sigma);
    geigs.init();
    int nconv = geigs.compute(SortRule::LargestMagn);
    if (geigs.info() != CompInfo::Successful or nconv < k) {
        // return what we have
        evals = std::vector<double>();
        evecs = Eigen::MatrixXd();
        return false;
    }
    Eigen::VectorXd ev = geigs.eigenvalues();
    const int nk = (int)ev.size();
    evals.resize(nk);
    for (int i = 0; i < nk; ++i) evals[i] = std::max(0.0, ev[i]); // clip negatives
    evecs = geigs.eigenvectors();
    // scipy eigsh returns eigenvalues in ascending order; sort ascending and reorder evecs columns
    std::vector<int> order(nk);
    for (int i = 0; i < nk; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return evals[a] < evals[b]; });
    std::vector<double> sorted_evals(nk);
    Eigen::MatrixXd sorted_evecs(evecs.rows(), nk);
    for (int i = 0; i < nk; ++i) {
        sorted_evals[i] = evals[order[i]];
        sorted_evecs.col(i) = evecs.col(order[i]);
    }
    evals.swap(sorted_evals);
    evecs.swap(sorted_evecs);
    return true;
}
// --- vertex normals (mesh: sum of face normals, normalized) ---
// NOTE: uses float32 accumulation to match the python version, which computes
// face normals in float32 torch and accumulates them (0.8% diff otherwise).
static std::vector<std::array<double,3>> mesh_vertex_normals(
    const std::vector<std::array<double,3>>& V,
    const std::vector<std::array<int32_t,3>>& F) {
    std::vector<std::array<float,3>> vn(V.size(), {0.0f,0.0f,0.0f});
    const float eps = 1e-6f;
    for (size_t f = 0; f < F.size(); ++f) {
        float a0=V[F[f][0]][0],a1=V[F[f][0]][1],a2=V[F[f][0]][2];
        float b0=V[F[f][1]][0],b1=V[F[f][1]][1],b2=V[F[f][1]][2];
        float c0=V[F[f][2]][0],c1=V[F[f][2]][1],c2=V[F[f][2]][2];
        float e10=b0-a0,e11=b1-a1,e12=b2-a2;
        float e20=c0-a0,e21=c1-a1,e22=c2-a2;
        float nx=e11*e22-e12*e21, ny=e12*e20-e10*e22, nz=e10*e21-e11*e20;
        float nn=std::sqrt(nx*nx+ny*ny+nz*nz)+eps;
        nx/=nn; ny/=nn; nz/=nn;
        for (int i=0;i<3;++i){ vn[F[f][i]][0]+=nx; vn[F[f][i]][1]+=ny; vn[F[f][i]][2]+=nz; }
    }
    std::vector<std::array<double,3>> out(V.size());
    for (size_t i=0;i<V.size();++i) {
        double x=vn[i][0],y=vn[i][1],z=vn[i][2];
        double nn=std::sqrt(x*x+y*y+z*z);
        if (nn>0){ out[i]={x/nn,y/nn,z/nn}; } else { out[i]={x,y,z}; }
    }
    return out;
}

// --- per-vertex tangent frames (matches build_tangent_frames) ---
std::vector<std::array<double,9>> build_tangent_frames(
    const std::vector<std::array<double,3>>& V,
    const std::vector<std::array<int32_t,3>>& F) {
    auto vn = mesh_vertex_normals(V, F);
    const double cand1[3] = {1,0,0};
    const double cand2[3] = {0,1,0};
    std::vector<std::array<double,9>> frames(V.size());
    for (size_t i = 0; i < V.size(); ++i) {
        const double* n = vn[i].data();
        // choose basisX candidate
        const double* cand = (std::abs(n[0]) < 0.9) ? cand1 : cand2;
        // project to tangent
        double d = cand[0]*n[0]+cand[1]*n[1]+cand[2]*n[2];
        double bx[3] = {cand[0]-d*n[0], cand[1]-d*n[1], cand[2]-d*n[2]};
        double bn = std::sqrt(bx[0]*bx[0]+bx[1]*bx[1]+bx[2]*bx[2]);
        if (bn>0){ bx[0]/=bn; bx[1]/=bn; bx[2]/=bn; }
        // basisY = cross(normal, basisX)
        double by[3]; cross3(n, bx, by);
        // store rows: X, Y, Z
        frames[i][0]=bx[0]; frames[i][1]=bx[1]; frames[i][2]=bx[2];
        frames[i][3]=by[0]; frames[i][4]=by[1]; frames[i][5]=by[2];
        frames[i][6]=n[0];  frames[i][7]=n[1];  frames[i][8]=n[2];
    }
    return frames;
}
// --- gradient operator (matches edge_tangent_vectors + build_grad) ---
void build_grad_ops(const std::vector<std::array<double,3>>& V,
                    const std::vector<std::array<int32_t,3>>& F,
                    Eigen::SparseMatrix<double>& gradX,
                    Eigen::SparseMatrix<double>& gradY) {
    const int nV = (int)V.size();
    // 1) L (needed for edges)
    Eigen::SparseMatrix<double> L = cotan_laplacian(V, F);
    // 2) frames
    auto frames = build_tangent_frames(V, F);

    // 3) edges from L's nonzeros (row,col)
    std::vector<int> e_row, e_col;
    for (int k = 0; k < L.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
            e_row.push_back((int)it.row());
            e_col.push_back((int)it.col());
        }
    const int nE = (int)e_row.size();

    // 4) edge tangent vectors (compX, compY) using tail vertex frame
    std::vector<double> compX(nE), compY(nE);
    for (int e = 0; e < nE; ++e) {
        int tail = e_row[e], tip = e_col[e];
        const double* t = V[tail].data();
        const double* p = V[tip].data();
        double ev[3] = {p[0]-t[0], p[1]-t[1], p[2]-t[2]};
        const double* bX = &frames[tail][0];
        const double* bY = &frames[tail][3];
        compX[e] = ev[0]*bX[0]+ev[1]*bX[1]+ev[2]*bX[2];
        compY[e] = ev[0]*bY[0]+ev[1]*bY[1]+ev[2]*bY[2];
    }

    // 5) build_grad: local least squares per vertex
    const double eps_reg = 1e-5;
    // outgoing edge list per vertex
    std::vector<std::vector<int>> vert_edge_out(nV);
    for (int e = 0; e < nE; ++e) {
        int tail = e_row[e], tip = e_col[e];
        if (tip != tail) vert_edge_out[tail].push_back(e);
    }

    std::vector<Triplet<std::complex<double>>> data_vals;
    for (int iV = 0; iV < nV; ++iV) {
        int n_neigh = (int)vert_edge_out[iV].size();
        if (n_neigh == 0) continue;
        // lhs_mat (n_neigh,2), rhs_mat (n_neigh, n_neigh+1)
        Eigen::MatrixXd lhs(n_neigh, 2);
        Eigen::MatrixXd rhs(n_neigh, n_neigh+1);
        rhs.setZero();
        std::vector<int> ind_lookup(n_neigh+1);
        ind_lookup[0] = iV;
        for (int in = 0; in < n_neigh; ++in) {
            int e = vert_edge_out[iV][in];
            int jV = e_col[e];
            ind_lookup[in+1] = jV;
            lhs(in,0) = 1.0 * compX[e];
            lhs(in,1) = 1.0 * compY[e];
            rhs(in,0) = 1.0 * (-1.0);
            rhs(in,in+1) = 1.0 * 1.0;
        }
        Eigen::MatrixXd lhs_T = lhs.transpose();
        Eigen::MatrixXd lhs_inv = (lhs_T * lhs + eps_reg * Eigen::MatrixXd::Identity(2,2)).inverse() * lhs_T;
        Eigen::MatrixXd sol = lhs_inv * rhs;   // (2, n_neigh+1)
        for (int in = 0; in < n_neigh+1; ++in) {
            int ig = ind_lookup[in];
            std::complex<double> c(sol(0,in), sol(1,in));
            data_vals.push_back({iV, ig, c});
        }
    }

    // build sparse complex matrix, split into real/imag
    Eigen::SparseMatrix<std::complex<double>> M(nV, nV);
    M.setFromTriplets(data_vals.begin(), data_vals.end());

    // split
    std::vector<Triplet<double>> tr, ti;
    for (int k = 0; k < M.outerSize(); ++k)
        for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(M, k); it; ++it) {
            tr.push_back({(int)it.row(), (int)it.col(), it.value().real()});
            ti.push_back({(int)it.row(), (int)it.col(), it.value().imag()});
        }
    gradX.resize(nV, nV); gradX.setFromTriplets(tr.begin(), tr.end());
    gradY.resize(nV, nV); gradY.setFromTriplets(ti.begin(), ti.end());
}
// --- normalize positions (matches normalize_positions(method='mean', scale_method='max_rad')) ---
// Center by the mean vertex position, then unit-scale by the max radius.
void normalize_positions(std::vector<std::array<double,3>>& V) {
    const size_t n = V.size();
    if (n == 0) return;
    double cx=0, cy=0, cz=0;
    for (const auto& p : V) { cx += p[0]; cy += p[1]; cz += p[2]; }
    cx /= (double)n; cy /= (double)n; cz /= (double)n;
    for (auto& p : V) { p[0]-=cx; p[1]-=cy; p[2]-=cz; }
    double maxr = 0.0;
    for (const auto& p : V) {
        double r = std::sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);
        if (r > maxr) maxr = r;
    }
    if (maxr > 0.0)
        for (auto& p : V) { p[0]/=maxr; p[1]/=maxr; p[2]/=maxr; }
}

// --- extract COO triplets (rows/cols int64, vals double) from a sparse matrix ---
// Matches scipy coo_matrix(M).row/.col/.data, used to feed the ONNX sparse-MM
// gather inputs (gx_rows/gx_cols/gx_vals, gy_rows/gy_cols/gy_vals).
void sparse_to_coo(const Eigen::SparseMatrix<double>& M,
                   std::vector<int64_t>& rows,
                   std::vector<int64_t>& cols,
                   std::vector<double>& vals) {
    rows.clear(); cols.clear(); vals.clear();
    rows.reserve(M.nonZeros()); cols.reserve(M.nonZeros()); vals.reserve(M.nonZeros());
    for (int k = 0; k < M.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
            rows.push_back((int64_t)it.row());
            cols.push_back((int64_t)it.col());
            vals.push_back(it.value());
        }
}
