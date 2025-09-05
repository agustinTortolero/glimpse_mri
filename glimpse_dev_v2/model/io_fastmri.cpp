#include "io_fastmri.hpp"

#include <H5public.h>
#include <H5Cpp.h>
#include <iostream>
#include <sstream>
#include <complex>
#include <vector>

// ---- OpenMP debug helper (top of file) ----
#if defined(_OPENMP)
#include <omp.h>
static inline int OMP_MAX_THREADS() { return omp_get_max_threads(); }
#else
#include <thread>
static inline int OMP_MAX_THREADS() {
    return static_cast<int>(std::thread::hardware_concurrency());
}
#endif


namespace {
template<class... Args>
void dbg_line(std::string* dbg, Args&&... a) {
    if (!dbg) return;
    std::ostringstream oss; (oss << ... << a);
    *dbg += oss.str(); *dbg += '\n';
}
} // anon

namespace mri {

// ----------- read reconstruction_rss (slice 0 if 3D) -----------
static void read_recon_rss_slice0(H5::H5File& file,
                                  std::vector<float>& out,
                                  int& ny, int& nx,
                                  std::string* dbg)
{
    if (!file.nameExists("reconstruction_rss")) { ny = nx = 0; return; }

    H5::DataSet ds = file.openDataSet("reconstruction_rss");
    H5::DataSpace sp = ds.getSpace();
    int rank = sp.getSimpleExtentNdims();
    std::vector<hsize_t> dims(rank);
    sp.getSimpleExtentDims(dims.data());

    if (rank == 3) {
        hsize_t S = dims[0];
        ny = (int)dims[1]; nx = (int)dims[2];
        dbg_line(dbg, "[io] reconstruction_rss dims=", S, " x ", ny, " x ", nx);
        if (S == 0) { ny = nx = 0; return; }

        out.resize((size_t)ny * nx);
        H5::DataSpace fs = ds.getSpace();
        hsize_t start[3] = {0,0,0}, count[3] = {1,(hsize_t)ny,(hsize_t)nx};
        fs.selectHyperslab(H5S_SELECT_SET, count, start);
        H5::DataSpace ms(3, count);
        ds.read(out.data(), H5::PredType::NATIVE_FLOAT, ms, fs);
    } else if (rank == 2) {
        ny = (int)dims[0]; nx = (int)dims[1];
        dbg_line(dbg, "[io] reconstruction_rss dims=", ny, " x ", nx);
        out.resize((size_t)ny * nx);
        ds.read(out.data(), H5::PredType::NATIVE_FLOAT);
    } else {
        dbg_line(dbg, "[io][WARN] reconstruction_rss rank=", rank, " not supported");
        ny = nx = 0;
    }
}

// ----------- float kspace with trailing 2 (real,imag) -----------
static bool read_kspace_float_tail2(H5::DataSet& ds, H5::DataSpace& sp,
                                    KSpace& ks, std::string* dbg)
{
    int rank = sp.getSimpleExtentNdims();
    std::vector<hsize_t> dims(rank);
    sp.getSimpleExtentDims(dims.data());

    auto read_one = [&](hsize_t C, hsize_t ny, hsize_t nx,
                        const std::vector<hsize_t>& start) -> bool
    {
        ks.coils = (int)C; ks.ny = (int)ny; ks.nx = (int)nx;
        const size_t N = (size_t)C * ny * nx;
        ks.host.resize(N);
        std::vector<float> tmp(N * 2); // [re,im]

        std::vector<hsize_t> count(rank);
        for (int i=0;i<rank;++i) count[i] = dims[i];

        H5::DataSpace fs = ds.getSpace();
        fs.selectHyperslab(H5S_SELECT_SET, count.data(), start.data());
        H5::DataSpace ms(rank, count.data());
        ds.read(tmp.data(), H5::PredType::NATIVE_FLOAT, ms, fs);

        // --- Parallel pack: tmp[{2*i,2*i+1}] -> ks.host[i] ---
        std::cerr << "[DBG][OMP][fastmri-tail2] N=" << (long long)N
                  << " threads=" << OMP_MAX_THREADS() << "\n";

        const long long Nll = static_cast<long long>(N);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long long i = 0; i < Nll; ++i) {
            const size_t ii = static_cast<size_t>(i);
            ks.host[ii] = std::complex<float>(tmp[2*ii + 0], tmp[2*ii + 1]);
        }

        dbg_line(dbg, "[io] Loaded float tail-2 kspace: C=", ks.coils, " ny=", ks.ny, " nx=", ks.nx);
        return true;
    };

    if (rank == 4 && dims[3] == 2) return read_one(dims[0], dims[1], dims[2], {0,0,0,0});
    if (rank == 5 && dims[4] == 2) return read_one(dims[1], dims[2], dims[3], {0,0,0,0,0});
    return false;
}

static bool read_kspace_compound(H5::DataSet& ds, H5::DataSpace& sp,
                                 KSpace& ks, std::string* dbg)
{
    int rank = sp.getSimpleExtentNdims();
    std::vector<hsize_t> dims(rank);
    sp.getSimpleExtentDims(dims.data());

    H5::DataType dt = ds.getDataType();
    if (dt.getClass() != H5T_COMPOUND) { dbg_line(dbg, "[io][ERR] Dataset not COMPOUND"); return false; }

    H5::CompType dsType(dt.getId());
    const int nmem = dsType.getNmembers();
    if (nmem < 2) { dbg_line(dbg, "[io][ERR] COMPOUND has <2 members (", nmem, ")"); return false; }

    // Build a matching memory type using dataset’s member names (often "real"/"imag")
    H5::CompType memType(sizeof(float)*2);
    {
        H5::DataType m0 = dsType.getMemberDataType(0);
        H5::DataType m1 = dsType.getMemberDataType(1);
        if (m0.getClass()!=H5T_FLOAT || m1.getClass()!=H5T_FLOAT) { dbg_line(dbg, "[io][ERR] members not FLOAT"); return false; }
        std::string name0 = dsType.getMemberName(0);
        std::string name1 = dsType.getMemberName(1);
        memType.insertMember(name0, 0,            H5::PredType::NATIVE_FLOAT);
        memType.insertMember(name1, sizeof(float),H5::PredType::NATIVE_FLOAT);
    }

    auto read_full3 = [&](hsize_t C, hsize_t ny, hsize_t nx) {
        ks.coils=(int)C; ks.ny=(int)ny; ks.nx=(int)nx;
        const size_t N=(size_t)C*ny*nx; ks.host.resize(N);
        std::vector<float> tmp(N*2);

        H5::DataSpace fs = ds.getSpace();
        std::vector<hsize_t> count(rank); for (int i=0;i<rank;++i) count[i]=dims[i];
        H5::DataSpace ms(rank, count.data());
        ds.read(tmp.data(), memType, ms, fs);

        std::cerr << "[DBG][OMP][fastmri-comp3] N=" << (long long)N
                  << " threads=" << OMP_MAX_THREADS() << "\n";

        const long long Nll = static_cast<long long>(N);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long long i = 0; i < Nll; ++i) {
            const size_t ii = static_cast<size_t>(i);
            ks.host[ii] = std::complex<float>(tmp[2*ii + 0], tmp[2*ii + 1]);
        }

        dbg_line(dbg, "[io] Loaded compound kspace: C=", ks.coils, " ny=", ks.ny, " nx=", ks.nx);
        return true;
    };

    auto read_slice4 = [&](hsize_t S, hsize_t C, hsize_t ny, hsize_t nx) {
        if (S==0) return false;
        ks.coils=(int)C; ks.ny=(int)ny; ks.nx=(int)nx;
        const size_t N=(size_t)C*ny*nx; ks.host.resize(N);
        std::vector<float> tmp(N*2);

        H5::DataSpace fs = ds.getSpace();
        hsize_t start[4] = {0,0,0,0}, count[4] = {1,C,ny,nx};
        fs.selectHyperslab(H5S_SELECT_SET, count, start);
        H5::DataSpace ms(4, count);
        ds.read(tmp.data(), memType, ms, fs);

        std::cerr << "[DBG][OMP][fastmri-comp4] N=" << (long long)N
                  << " threads=" << OMP_MAX_THREADS() << "\n";

        const long long Nll = static_cast<long long>(N);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long long i = 0; i < Nll; ++i) {
            const size_t ii = static_cast<size_t>(i);
            ks.host[ii] = std::complex<float>(tmp[2*ii + 0], tmp[2*ii + 1]);
        }

        dbg_line(dbg, "[io] Loaded compound rank-4 slice0: C=", ks.coils, " ny=", ks.ny, " nx=", ks.nx);
        return true;
    };

    if (rank==3) return read_full3(dims[0], dims[1], dims[2]);
    if (rank==4) return read_slice4(dims[0], dims[1], dims[2], dims[3]);
    dbg_line(dbg, "[io][WARN] compound rank=", rank, " not handled");
    return false;
}



// ----------- public loader -----------
bool load_fastmri_kspace(const std::string& path,
                         KSpace& ks,
                         std::vector<float>* preRecon,
                         int* preNy,
                         int* preNx,
                         std::string* dbg)
{
    try {
        dbg_line(dbg, "[io] open: ", path);
        H5::H5File file(path, H5F_ACC_RDONLY);

        int rss_h=0, rss_w=0;
        if (preRecon) {
            read_recon_rss_slice0(file, *preRecon, rss_h, rss_w, dbg);
            if (preNy) *preNy = rss_h;
            if (preNx) *preNx = rss_w;
        }

        if (file.nameExists("kspace")) {
            H5::DataSet ds = file.openDataSet("kspace");
            H5::DataType dt = ds.getDataType();
            H5T_class_t tcls = dt.getClass();
            H5::DataSpace sp = ds.getSpace();

            bool ok=false;
            if (tcls==H5T_FLOAT)    ok = read_kspace_float_tail2(ds, sp, ks, dbg);
            else if (tcls==H5T_COMPOUND) ok = read_kspace_compound(ds, sp, ks, dbg);
            else dbg_line(dbg, "[io][WARN] kspace typeclass=", (int)tcls, " not supported");

            if (ok) return true;

            if (preRecon && !preRecon->empty()) {
                dbg_line(dbg, "[io][WARN] kspace parse failed; using reconstruction_rss");
                ks.coils=1; ks.ny=rss_h; ks.nx=rss_w; // context for UI
                return true;
            }
            dbg_line(dbg, "[io][ERR] Unable to parse kspace");
            return false;
        } else {
            dbg_line(dbg, "[io][WARN] no 'kspace' dataset");
            if (preRecon && !preRecon->empty()) {
                ks.coils=1; ks.ny=rss_h; ks.nx=rss_w;
                return true;
            }
            return false;
        }
    } catch (const H5::Exception& e) {
        dbg_line(dbg, "[io][ERR] HDF5: ", e.getDetailMsg());
        return false;
    } catch (const std::exception& e) {
        dbg_line(dbg, "[io][ERR] ", e.what());
        return false;
    }
}

} // namespace mri
