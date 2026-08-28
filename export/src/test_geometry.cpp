// test_geometry.cpp
//
// Read a mesh (verts/faces) and compute geometric operators, then dump them to
// files so they can be compared against Python's get_operators() output.
//
// Usage: test_geometry <mesh.bin> <out_prefix>
// Outputs: <out_prefix>.mass, .L_diag, .evals, .evecs, .gradX, .gradY
//
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cstdint>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include "geometry_ops.h"

int main(int argc, char* argv[]) {
    std::string mesh_path = argc > 1 ? argv[1] : "test_mesh.bin";
    std::string out = argc > 2 ? argv[2] : "geo_out";

    // read mesh
    std::ifstream f(mesh_path, std::ios::binary);
    int64_t V, F;
    f.read((char*)&V, 8); f.read((char*)&F, 8);
    std::vector<std::array<double,3>> verts(V);
    std::vector<std::array<int32_t,3>> faces(F);
    f.read((char*)verts.data(), V*3*sizeof(double));
    f.read((char*)faces.data(), F*3*sizeof(int32_t));
    f.close();
    std::cout << "V=" << V << " F=" << F << std::endl;

    // 1) mass
    auto mass = vertex_areas(verts, faces);
    std::ofstream f_mass(out + ".mass");
    for (auto m : mass) f_mass << m << "\n";
    f_mass.close();

    // 2) Laplacian (diag + a few off-diag for checking)
    auto L = cotan_laplacian(verts, faces);
    std::ofstream f_L(out + ".L");
    f_L << "V " << V << "\n";
    for (int k = 0; k < L.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
            f_L << it.row() << " " << it.col() << " " << it.value() << "\n";
    f_L.close();

    // 3) eigendecomposition
    double eps = 1e-8;
    std::vector<double> mass_eig = mass;
    double mean = 0; for (auto m : mass) mean += m; mean /= mass.size();
    for (auto& m : mass_eig) m += eps * mean;   // massvec += eps*mean(massvec)
    int k = 10;   // small for test
    std::vector<double> evals;
    Eigen::MatrixXd evecs;
    bool ok = generalized_eigs(L, mass_eig, k, eps, eps, evals, evecs);
    std::cout << "eig ok=" << ok << " nevals=" << evals.size() << std::endl;
    std::ofstream f_eval(out + ".evals");
    for (auto e : evals) f_eval << e << "\n";
    f_eval.close();
    std::ofstream f_evec(out + ".evecs");
    for (int i = 0; i < evecs.rows() and i < 5; ++i) {
        for (int j = 0; j < evecs.cols(); ++j) f_evec << evecs(i,j) << " ";
        f_evec << "\n";
    }
    f_evec.close();

    // 4) gradient operators
    Eigen::SparseMatrix<double> gX, gY;
    build_grad_ops(verts, faces, gX, gY);
    std::ofstream f_gx(out + ".gradX");
    for (int kk = 0; kk < gX.outerSize(); ++kk)
        for (Eigen::SparseMatrix<double>::InnerIterator it(gX, kk); it; ++it)
            if (std::abs(it.value()) > 1e-12) f_gx << it.row() << " " << it.col() << " " << it.value() << "\n";
    f_gx.close();

    std::cout << "done. outputs -> " << out << ".*" << std::endl;
    return 0;
}
