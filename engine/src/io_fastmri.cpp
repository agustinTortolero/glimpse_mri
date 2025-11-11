#include "io_fastmri.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>

// ---- Feature flag (override in build if needed) -----------------------------
#if !defined(ENGINE_HAS_FASTMRI)
#define ENGINE_HAS_FASTMRI 1
#endif

#if ENGINE_HAS_FASTMRI
#include <hdf5.h>
#endif

// ============================================================================
// When fastMRI is disabled at build time, provide stubs that log and fail
// ============================================================================
#if !ENGINE_HAS_FASTMRI

bool fastmri_probe(const std::string& path, FastMRIInfo& info) {
    (void)path; (void)info;
    dbg_head("fastMRI"); std::cerr << "fastMRI disabled at build.\n";
    return false;
}

bool fastmri_load_rss_slice(const std::string& path, int slice_idx,
    std::vector<float>& rss, int& ny, int& nx) {
    (void)path; (void)slice_idx; (void)rss; (void)ny; (void)nx;
    dbg_head("fastMRI"); std::cerr << "fastMRI disabled at build.\n";
    return false;
}

bool fastmri_load_kspace_slice(const std::string& path, int slice_idx, KsGrid& ks) {
    (void)path; (void)slice_idx; (void)ks;
    dbg_head("fastMRI"); std::cerr << "fastMRI disabled at build.\n";
    return false;
}

#else // ENGINE_HAS_FASTMRI

// ============================================================================
// HDF5 helpers (RAII)
// ============================================================================
struct H5Closer {
    hid_t id = -1;
    enum Kind { KNone, KFile, KDataset, KSpace, KType, KProp } kind = KNone;
    H5Closer() = default;
    H5Closer(hid_t i, Kind k) : id(i), kind(k) {}
    ~H5Closer() { close(); }
    void reset(hid_t i = -1, Kind k = KNone) { close(); id = i; kind = k; }
    void close() {
        if (id < 0) return;
        switch (kind) {
        case KFile:    H5Fclose(id); break;
        case KDataset: H5Dclose(id); break;
        case KSpace:   H5Sclose(id); break;
        case KType:    H5Tclose(id); break;
        case KProp:    H5Pclose(id); break;
        default: break;
        }
        id = -1; kind = KNone;
    }
};

static bool dataset_exists(hid_t file, const char* name) {
    htri_t e = H5Lexists(file, name, H5P_DEFAULT);
    return (e > 0);
}

static bool get_dims(hid_t dset, std::vector<hsize_t>& dims_out) {
    H5Closer sp(H5Dget_space(dset), H5Closer::KSpace);
    if (sp.id < 0) return false;
    int rank = H5Sget_simple_extent_ndims(sp.id);
    if (rank <= 0 || rank > 8) return false;
    dims_out.resize(rank);
    H5Sget_simple_extent_dims(sp.id, dims_out.data(), nullptr);
    return true;
}

static bool is_compound_complex64(hid_t dset, std::string& name0, std::string& name1) {
    H5Closer dt(H5Dget_type(dset), H5Closer::KType);
    if (dt.id < 0) return false;
    if (H5Tget_class(dt.id) != H5T_COMPOUND) return false;
    const int nmem = H5Tget_nmembers(dt.id);
    if (nmem != 2) return false;

    hid_t m0 = H5Tget_member_type(dt.id, 0);
    hid_t m1 = H5Tget_member_type(dt.id, 1);
    bool ok = (H5Tget_class(m0) == H5T_FLOAT) && (H5Tget_class(m1) == H5T_FLOAT) &&
        (H5Tget_size(m0) == sizeof(float)) && (H5Tget_size(m1) == sizeof(float));

    if (ok) {
        char* n0 = H5Tget_member_name(dt.id, 0);
        char* n1 = H5Tget_member_name(dt.id, 1);
        if (n0) { name0 = n0; H5free_memory(n0); }
        if (n1) { name1 = n1; H5free_memory(n1); }
    }
    H5Tclose(m0); H5Tclose(m1);
    return ok;
}

// Layout type for reading compound datasets
struct H5Cpx { float r, i; };

// ============================================================================
// Probe
// ============================================================================
bool fastmri_probe(const std::string& path, FastMRIInfo& info) {
    dbg_head("fastMRI"); std::cerr << "probe: '" << path << "'\n";

    H5Closer f(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Closer::KFile);
    if (f.id < 0) {
        dbg_head("fastMRI"); std::cerr << "open failed\n";
        return false;
    }

    const bool kspace_ok = dataset_exists(f.id, "kspace");
    const bool rss_ok = dataset_exists(f.id, "reconstruction_rss");

    if (!kspace_ok && !rss_ok) {
        dbg_head("fastMRI"); std::cerr << "no 'kspace' nor 'reconstruction_rss' dataset\n";
        return false;
    }

    if (kspace_ok) {
        H5Closer d(H5Dopen2(f.id, "kspace", H5P_DEFAULT), H5Closer::KDataset);
        if (d.id < 0) return false;

        std::vector<hsize_t> dims;
        if (!get_dims(d.id, dims)) return false;

        std::string n0, n1;
        const bool is_comp = is_compound_complex64(d.id, n0, n1);

        if (is_comp) {
            // expected dims: [slices, coils, ny, nx]
            if (dims.size() != 4) {
                dbg_head("fastMRI"); std::cerr << "kspace compound rank=" << dims.size() << " (expect 4)\n";
                return false;
            }
            info.slices = (int)dims[0];
            info.coils = (int)dims[1];
            info.ny = (int)dims[2];
            info.nx = (int)dims[3];
        }
        else {
            // Sometimes stored as float32 with last dim=2 -> [slices, coils, ny, nx, 2]
            if (dims.size() != 5 || dims.back() != 2) {
                dbg_head("fastMRI"); std::cerr << "kspace not recognized dtype/rank\n";
                return false;
            }
            info.slices = (int)dims[0];
            info.coils = (int)dims[1];
            info.ny = (int)dims[2];
            info.nx = (int)dims[3];
        }
        dbg_head("fastMRI"); std::cerr << "kspace dims: S=" << info.slices
            << " C=" << info.coils
            << " ny=" << info.ny
            << " nx=" << info.nx << "\n";
    }

    if (rss_ok) {
        H5Closer d(H5Dopen2(f.id, "reconstruction_rss", H5P_DEFAULT), H5Closer::KDataset);
        if (d.id >= 0) {
            std::vector<hsize_t> dims;
            if (get_dims(d.id, dims) && dims.size() == 3) {
                info.slices = std::max(info.slices, (int)dims[0]);
                info.ny = std::max(info.ny, (int)dims[1]);
                info.nx = std::max(info.nx, (int)dims[2]);
                info.has_rss = true;
                dbg_head("fastMRI"); std::cerr << "RSS dims: S=" << dims[0]
                    << " ny=" << dims[1]
                    << " nx=" << dims[2] << "\n";
            }
        }
    }

    // Optional switch to ignore precomputed RSS if you want to force recon.
#if FASTMRI_IGNORE_RSS
    if (info.has_rss) {
        dbg_head("fastMRI"); std::cerr << "IGNORE_RSS=1 → forcing reconstruct path (pretend no RSS).\n";
        info.has_rss = false;
    }
#endif

    // Minimal sanity
    if (info.slices <= 0 || (info.coils <= 0 && !info.has_rss)) {
        dbg_head("fastMRI"); std::cerr << "probe sanity failed (S=" << info.slices
            << " C=" << info.coils
            << " has_rss=" << (info.has_rss ? 1 : 0) << ")\n";
        return false;
    }
    return true;
}

// ============================================================================
// Load RSS slice
// ============================================================================
bool fastmri_load_rss_slice(const std::string& path, int slice_idx,
    std::vector<float>& rss, int& ny, int& nx)
{
    H5Closer f(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Closer::KFile);
    if (f.id < 0) return false;

    H5Closer d(H5Dopen2(f.id, "reconstruction_rss", H5P_DEFAULT), H5Closer::KDataset);
    if (d.id < 0) return false;

    std::vector<hsize_t> dims;
    if (!get_dims(d.id, dims) || dims.size() != 3) return false;

    const int slices = (int)dims[0];
    ny = (int)dims[1];
    nx = (int)dims[2];
    if (slice_idx < 0 || slice_idx >= slices) return false;

    // Select hyperslab [slice_idx, :, :]
    H5Closer space(H5Dget_space(d.id), H5Closer::KSpace);
    hsize_t start[3] = { (hsize_t)slice_idx, 0, 0 };
    hsize_t count[3] = { 1, (hsize_t)ny, (hsize_t)nx };
    if (H5Sselect_hyperslab(space.id, H5S_SELECT_SET, start, nullptr, count, nullptr) < 0) return false;

    // Memory space
    hsize_t mdims[3] = { 1, (hsize_t)ny, (hsize_t)nx };
    H5Closer mspace(H5Screate_simple(3, mdims, nullptr), H5Closer::KSpace);

    rss.resize((size_t)ny * (size_t)nx);
    if (H5Dread(d.id, H5T_NATIVE_FLOAT, mspace.id, space.id, H5P_DEFAULT, rss.data()) < 0) return false;

    dbg_head("fastMRI"); std::cerr << "Loaded RSS slice=" << slice_idx << " (" << nx << "x" << ny << ")\n";
    return true;
}

// ============================================================================
// Load k-space slice
// ============================================================================
static bool read_kspace_compound_slice(hid_t dset, int slice_idx,
    int coils, int ny, int nx,
    std::vector<H5Cpx>& buf_out)
{
    H5Closer space(H5Dget_space(dset), H5Closer::KSpace);
    if (space.id < 0) return false;

    // Select [slice_idx, 0..coils-1, 0..ny-1, 0..nx-1]
    hsize_t start[4] = { (hsize_t)slice_idx, 0, 0, 0 };
    hsize_t count[4] = { 1, (hsize_t)coils, (hsize_t)ny, (hsize_t)nx };
    if (H5Sselect_hyperslab(space.id, H5S_SELECT_SET, start, nullptr, count, nullptr) < 0) return false;

    // Memory space with same shape
    hsize_t mdims[4] = { 1, (hsize_t)coils, (hsize_t)ny, (hsize_t)nx };
    H5Closer mspace(H5Screate_simple(4, mdims, nullptr), H5Closer::KSpace);

    // Create memory compound type matching dataset member names
    std::string n0, n1;
    if (!is_compound_complex64(dset, n0, n1)) return false;
    H5Closer mtype(H5Tcreate(H5T_COMPOUND, sizeof(H5Cpx)), H5Closer::KType);
    H5Tinsert(mtype.id, n0.c_str(), HOFFSET(H5Cpx, r), H5T_NATIVE_FLOAT);
    H5Tinsert(mtype.id, n1.c_str(), HOFFSET(H5Cpx, i), H5T_NATIVE_FLOAT);

    buf_out.resize((size_t)coils * ny * nx);
    if (H5Dread(dset, mtype.id, mspace.id, space.id, H5P_DEFAULT, buf_out.data()) < 0) return false;
    return true;
}

static bool read_kspace_splitri_slice(hid_t dset, int slice_idx,
    int coils, int ny, int nx,
    std::vector<float>& buf_ri_out)
{
    // Dataset is float32 with last dimension 2, rank=5: [S,C,ny,nx,2]
    H5Closer space(H5Dget_space(dset), H5Closer::KSpace);
    if (space.id < 0) return false;

    hsize_t start[5] = { (hsize_t)slice_idx, 0, 0, 0, 0 };
    hsize_t count[5] = { 1, (hsize_t)coils, (hsize_t)ny, (hsize_t)nx, 2 };
    if (H5Sselect_hyperslab(space.id, H5S_SELECT_SET, start, nullptr, count, nullptr) < 0) return false;

    hsize_t mdims[5] = { 1, (hsize_t)coils, (hsize_t)ny, (hsize_t)nx, 2 };
    H5Closer mspace(H5Screate_simple(5, mdims, nullptr), H5Closer::KSpace);

    buf_ri_out.resize((size_t)coils * ny * nx * 2);
    if (H5Dread(dset, H5T_NATIVE_FLOAT, mspace.id, space.id, H5P_DEFAULT, buf_ri_out.data()) < 0) return false;
    return true;
}

bool fastmri_load_kspace_slice(const std::string& path, int slice_idx, KsGrid& ks)
{
    H5Closer f(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Closer::KFile);
    if (f.id < 0) return false;

    H5Closer d(H5Dopen2(f.id, "kspace", H5P_DEFAULT), H5Closer::KDataset);
    if (d.id < 0) return false;

    // Shape
    std::vector<hsize_t> dims;
    if (!get_dims(d.id, dims)) return false;

    bool compound = false;
    if (dims.size() == 4) {
        std::string n0, n1;
        compound = is_compound_complex64(d.id, n0, n1);
        if (!compound) {
            dbg_head("fastMRI"); std::cerr << "rank=4 but not compound complex64\n";
            return false;
        }
    }
    else if (dims.size() == 5) {
        if (dims.back() != 2) {
            dbg_head("fastMRI"); std::cerr << "rank=5 but last dim != 2\n";
            return false;
        }
        compound = false;
    }
    else {
        dbg_head("fastMRI"); std::cerr << "unsupported kspace rank=" << dims.size() << "\n";
        return false;
    }

    const int slices = (int)dims[0];
    const int coils = (int)dims[1];
    const int ny = (int)dims[2];
    const int nx = (int)dims[3];
    if (slice_idx < 0 || slice_idx >= slices) return false;

    ks.coils = coils; ks.ny = ny; ks.nx = nx;
    ks.host.assign((size_t)coils * ny * nx, std::complex<float>(0, 0));

    if (compound) {
        std::vector<H5Cpx> tmp;  // flattened [1,C,ny,nx]
        if (!read_kspace_compound_slice(d.id, slice_idx, coils, ny, nx, tmp)) return false;

        size_t p = 0;
        for (int c = 0; c < coils; ++c) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x, ++p) {
                    ks.host[((size_t)c * ny + y) * nx + x] = std::complex<float>(tmp[p].r, tmp[p].i);
                }
            }
        }
    }
    else {
        std::vector<float> tmp; // flattened [1,C,ny,nx,2]
        if (!read_kspace_splitri_slice(d.id, slice_idx, coils, ny, nx, tmp)) return false;

        size_t p = 0;
        for (int c = 0; c < coils; ++c) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    float r = tmp[p++], i = tmp[p++];
                    ks.host[((size_t)c * ny + y) * nx + x] = std::complex<float>(r, i);
                }
            }
        }
    }

    dbg_head("fastMRI"); std::cerr << "Loaded kspace slice=" << slice_idx
        << " C=" << coils << " ny=" << ny << " nx=" << nx << "\n";
    return true;
}

#endif // ENGINE_HAS_FASTMRI
