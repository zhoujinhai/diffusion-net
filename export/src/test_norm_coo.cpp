#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cstdint>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include "geometry_ops.h"

int main(int argc, char* argv[]) {
    std::string mesh_path = argc > 1 ? argv[1] : "test_raw.bin";
    std::string out = argc > 2 ? argv[2] : "cpp_out";

    std::ifstream f(mesh_path, std::ios::binary);
    int64_t V, F;
    f.read((char*)&V, 8); f.read((char*)&F, 8);
    std::vector<std::array<double,3>> verts(V);
    std::vector<std::array<int32_t,3>> faces(F);
    f.read((char*)verts.data(), V*3*sizeof(double));
    f.read((char*)faces.data(), F*3*sizeof(int32_t));
    f.close();

    // 1) normalize positions first (matches get_operators pipeline)
    normalize_positions(verts);
    std::ofstream fo(out + ".verts_norm", std::ios::binary);
    fo.write((char*)verts.data(), V*3*sizeof(double));
    fo.close();

    // 2) dump tangent frames from normalized verts (for comparison)
    auto frames = build_tangent_frames(verts, faces);
    std::ofstream ff(out + ".frames", std::ios::binary);
    ff.write((char*)frames.data(), V*9*sizeof(double));
    ff.close();

    // 3) gradient ops + COO extraction
    Eigen::SparseMatrix<double> gX, gY;
    build_grad_ops(verts, faces, gX, gY);
    std::vector<int64_t> gxr, gxc, gyr, gyc;
    std::vector<double> gxv, gyv;
    sparse_to_coo(gX, gxr, gxc, gxv);
    sparse_to_coo(gY, gyr, gyc, gyv);

    auto write_coo = [&](const std::string& pfx,
                         const std::vector<int64_t>& r, const std::vector<int64_t>& c,
                         const std::vector<double>& v) {
        std::ofstream o(out + "." + pfx, std::ios::binary);
        int64_t nnz = (int64_t)r.size();
        o.write((char*)&nnz, 8);
        o.write((char*)r.data(), nnz*8);
        o.write((char*)c.data(), nnz*8);
        o.write((char*)v.data(), nnz*8);
        o.close();
    };
    write_coo("gx", gxr, gxc, gxv);
    write_coo("gy", gyr, gyc, gyv);
    std::cout << "V=" << V << " NX=" << gxr.size() << " NY=" << gyr.size() << std::endl;
    std::cout << "done -> " << out << ".*" << std::endl;
    return 0;
}

