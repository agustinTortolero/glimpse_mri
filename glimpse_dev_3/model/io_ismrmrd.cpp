#include "io_ismrmrd.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <complex>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#pragma warning(push)
#pragma warning(disable:4251) // quiet HDF5 dll-interface warnings in MSVC
#include <H5public.h>
#include <H5Cpp.h>
#include <H5Epublic.h>
#pragma warning(pop)

// ISMRMRD (via vcpkg)
#include <ismrmrd/dataset.h>
#include <ismrmrd/ismrmrd.h>
#include <ismrmrd/xml.h>


// ---- OpenMP debug helper ----
#if defined(_OPENMP)
#include <omp.h>
static inline int OMP_MAX_THREADS() { return omp_get_max_threads(); }
#else
#include <thread>
static inline int OMP_MAX_THREADS() {
    return static_cast<int>(std::thread::hardware_concurrency());
}
#endif


// -----------------------------------------------------------------------------
// helpers (anonymous namespace) — prototypes first for clarity
// -----------------------------------------------------------------------------
namespace {

// dbg collector
template<class... Args> void dbg_line(std::string* dbg, Args&&... a);

// HDF5 error-stack suppression (call once)
static void suppress_hdf5_error_stack_once();

// tiny string helper
static std::string find_between(const std::string&, const std::string&, const std::string&);

// xml readers
static void        try_read_ismrmrd_xml(H5::H5File&, std::string* dbg);
static std::string read_xml_text(H5::H5File&);

// header parsing + guards
static const char* traj_to_str(ISMRMRD::TrajectoryType t);
static void clamp_y_limits(int ny_hdr, int& ymin, int& ycenter, int& ymax, std::string* dbg);
static bool parse_header_dims(const std::string& xml,
                              int& nx_hdr, int& ny_hdr,
                              int& ymin, int& ycenter, int& ymax,
                              std::string& traj,
                              std::string* dbg);

// image readers
static bool read_float_2D(H5::DataSet&, int& ny, int& nx, std::vector<float>&, std::string* dbg);
static bool read_compound_2D_mag(H5::DataSet&, int& ny, int& nx, std::vector<float>&, std::string* dbg);
static bool read_tail2_2Dor3D_mag_slice0(H5::DataSet&, int& ny, int& nx, std::vector<float>&, std::string* dbg);
static bool try_read_any_image(H5::H5File&, std::vector<float>&, int& ny, int& nx, std::string* dbg);

// k-space readers (dense datasets)
static bool read_kspace_float_tail2(H5::DataSet&, H5::DataSpace&, mri::KSpace&, std::string* dbg);
static bool read_kspace_compound(H5::DataSet&, H5::DataSpace&, mri::KSpace&, std::string* dbg);
static bool try_read_any_kspace(H5::H5File&, mri::KSpace&, std::string* dbg);

// assemble from /dataset/acquisitions (Cartesian only)
static bool assemble_from_acquisitions(const std::string& path,
                                       const std::string& xml,
                                       mri::KSpace& ks,
                                       std::string* dbg);

// -----------------------------------------------------------------------------
// definitions
// -----------------------------------------------------------------------------

template<class... Args>
void dbg_line(std::string* dbg, Args&&... a) {
    if (!dbg) return;
    std::ostringstream oss; (oss << ... << a);
    *dbg += oss.str(); *dbg += '\n';
}

static void suppress_hdf5_error_stack_once() {
    static bool done=false; if (done) return; done=true;
    H5::Exception::dontPrint();                 // C++ API
    H5Eset_auto(H5E_DEFAULT, nullptr, nullptr); // C API
}

static std::string find_between(const std::string& s,
                                const std::string& a, const std::string& b) {
    size_t i = s.find(a); if (i == std::string::npos) return "";
    i += a.size();
    size_t j = s.find(b, i); if (j == std::string::npos) return "";
    return s.substr(i, j - i);
}

static void try_read_ismrmrd_xml(H5::H5File& f, std::string* dbg) {
    try {
        if (!f.nameExists("/dataset/xml")) { dbg_line(dbg, "[ismrmrd] no /dataset/xml"); return; }
        H5::DataSet ds = f.openDataSet("/dataset/xml");
        H5::StrType mem(H5::PredType::C_S1, H5T_VARIABLE);
        char* cxml = nullptr; ds.read(&cxml, mem);
        std::string xml = cxml ? std::string(cxml) : std::string();
        if (cxml) free(cxml);

        dbg_line(dbg, "[ismrmrd] xml bytes = ", xml.size());
        std::string enc   = find_between(xml, "<encoding>", "</encoding>");
        std::string mX    = find_between(enc, "<x>", "</x>");
        std::string mY    = find_between(enc, "<y>", "</y>");
        std::string mZ    = find_between(enc, "<z>", "</z>");
        std::string traj  = find_between(enc, "<trajectory>", "</trajectory>");
        if (!mX.empty() || !mY.empty() || !mZ.empty() || !traj.empty()) {
            dbg_line(dbg, "[ismrmrd][meta] encoded matrix = [",
                     (mX.empty()?"?":mX), ",",
                     (mY.empty()?"?":mY), ",",
                     (mZ.empty()?"?":mZ), "]");
            dbg_line(dbg, "[ismrmrd][meta] trajectory = ", (traj.empty()?"?":traj));
        }
    } catch (const H5::Exception& e) {
        dbg_line(dbg, "[ismrmrd][xml][ERR] HDF5: ", e.getDetailMsg());
    }
}

static std::string read_xml_text(H5::H5File& f) {
    try {
        if (!f.nameExists("/dataset/xml")) return {};
        H5::DataSet ds = f.openDataSet("/dataset/xml");
        H5::StrType mem(H5::PredType::C_S1, H5T_VARIABLE);
        char* cxml = nullptr; ds.read(&cxml, mem);
        std::string xml = cxml ? std::string(cxml) : std::string();
        if (cxml) free(cxml);
        return xml;
    } catch (...) { return {}; }
}

static const char* traj_to_str(ISMRMRD::TrajectoryType t) {
    using TT = ISMRMRD::TrajectoryType;
    switch (t) {
    case TT::CARTESIAN:   return "cartesian";
    case TT::EPI:         return "epi";
    case TT::RADIAL:      return "radial";
    case TT::GOLDENANGLE: return "goldenangle";
    case TT::SPIRAL:      return "spiral";
    case TT::OTHER:       return "other";
    default:              return "unknown";
    }
}

static void clamp_y_limits(int ny_hdr, int& ymin, int& ycenter, int& ymax, std::string* dbg) {
    if (ny_hdr > 0 && ymax >= ny_hdr) {
        dbg_line(dbg, "[ismrmrd][hdr][FIX] ymax(", ymax, ") >= ny_hdr(", ny_hdr, ") -> clamp to ", ny_hdr-1);
        ymax = ny_hdr - 1;
    }
    if (ymin < 0) {
        dbg_line(dbg, "[ismrmrd][hdr][FIX] ymin(", ymin, ") < 0 -> clamp to 0");
        ymin = 0;
    }
    if (ymin > ymax && ny_hdr > 0) {
        dbg_line(dbg, "[ismrmrd][hdr][FIX] ymin(", ymin, ") > ymax(", ymax, ") -> reset to [0,", ny_hdr-1, "]");
        ymin = 0; ymax = ny_hdr - 1;
    }
    if (ycenter < ymin || ycenter > ymax) {
        const int old = ycenter;
        ycenter = (ymin + ymax) / 2;
        dbg_line(dbg, "[ismrmrd][hdr][FIX] ycenter(", old, ") outside [", ymin, ",", ymax, "] -> ", ycenter);
    }
}

static bool parse_header_dims(const std::string& xml,
                              int& nx_hdr, int& ny_hdr,
                              int& ymin, int& ycenter, int& ymax,
                              std::string& traj,
                              std::string* dbg)
{
    nx_hdr = 0; ny_hdr = 0; ymin = 0; ycenter = 0; ymax = -1; traj = "unknown";

    // Preferred: formal parse
    try {
        ISMRMRD::IsmrmrdHeader hdr;
        ISMRMRD::deserialize(xml.c_str(), hdr);
        if (hdr.encoding.empty()) throw std::runtime_error("no <encoding>");
        const auto& enc = hdr.encoding[0];

        nx_hdr = (int)enc.encodedSpace.matrixSize.x;
        ny_hdr = (int)enc.encodedSpace.matrixSize.y;
        traj   = traj_to_str(enc.trajectory);

        if (enc.encodingLimits.kspace_encoding_step_1.is_present()) {
            const auto& lim = enc.encodingLimits.kspace_encoding_step_1.get();
            ymin    = (int)lim.minimum;
            ymax    = (int)lim.maximum; // inclusive
            ycenter = (int)lim.center;
        } else {
            ymin = 0; ymax = ny_hdr>0 ? ny_hdr-1 : 0; ycenter = (ymin + ymax) / 2;
        }

        clamp_y_limits(ny_hdr, ymin, ycenter, ymax, dbg);
        dbg_line(dbg, "[ismrmrd][meta] encoded matrix = [", nx_hdr, ",", ny_hdr, ",?]");
        dbg_line(dbg, "[ismrmrd][meta] trajectory = ", traj);
        dbg_line(dbg, "[ismrmrd][hdr] y[min,center,max] = [", ymin, ",", ycenter, ",", ymax, "]");
        return true;
    } catch (const std::exception& e) {
        dbg_line(dbg, "[ismrmrd][hdr][WARN] lib parse failed: ", e.what(), " -> trying naive parse...");
    }

    // Fallback: naive string scan
    std::string enc   = find_between(xml, "<encoding>", "</encoding>");
    std::string sx    = find_between(enc, "<x>", "</x>");
    std::string sy    = find_between(enc, "<y>", "</y>");
    std::string tr    = find_between(enc, "<trajectory>", "</trajectory>");

    if (!sx.empty()) nx_hdr = std::atoi(sx.c_str());
    if (!sy.empty()) ny_hdr = std::atoi(sy.c_str());
    traj = tr.empty() ? "unknown" : tr;

    std::string lim   = find_between(enc, "<encodingLimits>", "</encodingLimits>");
    std::string step1 = find_between(lim, "<kspace_encoding_step_1>", "</kspace_encoding_step_1>");
    std::string smin  = find_between(step1, "<minimum>", "</minimum>");
    std::string smax  = find_between(step1, "<maximum>", "</maximum>");
    std::string scen  = find_between(step1, "<center>",  "</center>");

    if (!smin.empty()) ymin = std::atoi(smin.c_str());
    if (!smax.empty()) ymax = std::atoi(smax.c_str());
    if (!scen.empty()) ycenter = std::atoi(scen.c_str());

    if (smin.empty() || smax.empty()) { ymin = 0; ymax = ny_hdr>0 ? ny_hdr-1 : 0; ycenter = (ymin + ymax) / 2; }
    else if (scen.empty()) { ycenter = (ymin + ymax) / 2; }

    clamp_y_limits(ny_hdr, ymin, ycenter, ymax, dbg);
    dbg_line(dbg, "[ismrmrd][meta][naive] encoded matrix = [", nx_hdr, ",", ny_hdr, ",?]");
    dbg_line(dbg, "[ismrmrd][meta][naive] trajectory = ", traj);
    dbg_line(dbg, "[ismrmrd][hdr][naive] y[min,center,max] = [", ymin, ",", ycenter, ",", ymax, "]");
    return (nx_hdr > 0 && ny_hdr > 0);
}

// ---- image readers ----
static bool read_float_2D(H5::DataSet& ds, int& ny, int& nx,
                          std::vector<float>& out, std::string* dbg) {
    H5::DataSpace sp = ds.getSpace();
    if (sp.getSimpleExtentNdims() != 2) return false;
    hsize_t dims[2]; sp.getSimpleExtentDims(dims);
    ny = (int)dims[0]; nx = (int)dims[1];
    out.resize((size_t)ny*nx);
    ds.read(out.data(), H5::PredType::NATIVE_FLOAT);
    dbg_line(dbg, "[ismrmrd][img] float[", ny, "x", nx, "]");
    return true;
}

static bool read_compound_2D_mag(H5::DataSet& ds, int& ny, int& nx,
                                 std::vector<float>& out, std::string* dbg) {
    H5::DataType dt = ds.getDataType();
    if (dt.getClass() != H5T_COMPOUND) return false;
    H5::DataSpace sp = ds.getSpace();
    if (sp.getSimpleExtentNdims() != 2) return false;

    hsize_t dims[2]; sp.getSimpleExtentDims(dims);
    ny = (int)dims[0]; nx = (int)dims[1];

    H5::CompType dsType(dt.getId());
    if (dsType.getNmembers() < 2) return false;

    H5::CompType memType(sizeof(float)*2);
    memType.insertMember(dsType.getMemberName(0), 0,             H5::PredType::NATIVE_FLOAT);
    memType.insertMember(dsType.getMemberName(1), sizeof(float), H5::PredType::NATIVE_FLOAT);

    std::vector<float> tmp((size_t)ny*nx*2);
    ds.read(tmp.data(), memType);
    out.resize((size_t)ny*nx);
    for (size_t i=0;i<out.size();++i) {
        float re = tmp[2*i+0], im = tmp[2*i+1];
        out[i] = std::sqrt(re*re + im*im);
    }
    dbg_line(dbg, "[ismrmrd][img] compound->mag[", ny, "x", nx, "]");
    return true;
}

static bool read_tail2_2Dor3D_mag_slice0(H5::DataSet& ds, int& ny, int& nx,
                                         std::vector<float>& out, std::string* dbg) {
    H5::DataSpace sp = ds.getSpace();
    int rank = sp.getSimpleExtentNdims();
    if (rank != 3 && rank != 4) return false;

    std::vector<hsize_t> dims(rank);
    sp.getSimpleExtentDims(dims.data());
    if (dims.back() != 2) return false;

    if (rank == 3) {
        ny = (int)dims[0]; nx = (int)dims[1];
        const size_t N = (size_t)ny*nx;
        std::vector<float> tmp(N*2);
        ds.read(tmp.data(), H5::PredType::NATIVE_FLOAT);
        out.resize(N);
        for (size_t i=0;i<N;++i) {
            float re = tmp[2*i+0], im = tmp[2*i+1];
            out[i] = std::sqrt(re*re + im*im);
        }
        dbg_line(dbg, "[ismrmrd][img] tail2 3D->mag[", ny, "x", nx, "]");
        return true;
    } else {
        hsize_t S = dims[0]; if (S == 0) return false;
        ny = (int)dims[1]; nx = (int)dims[2];
        const size_t N = (size_t)ny*nx;
        std::vector<float> tmp(N*2);
        H5::DataSpace fs = ds.getSpace();
        hsize_t start[4] = {0,0,0,0}, count[4] = {1,(hsize_t)ny,(hsize_t)nx,2};
        fs.selectHyperslab(H5S_SELECT_SET, count, start);
        H5::DataSpace ms(4, count);
        ds.read(tmp.data(), H5::PredType::NATIVE_FLOAT, ms, fs);
        out.resize(N);
        for (size_t i=0;i<N;++i) {
            float re = tmp[2*i+0], im = tmp[2*i+1];
            out[i] = std::sqrt(re*re + im*im);
        }
        dbg_line(dbg, "[ismrmrd][img] tail2 4D slice0->mag[", ny, "x", nx, "]");
        return true;
    }
}

static bool try_read_any_image(H5::H5File& f,
                               std::vector<float>& out, int& ny, int& nx,
                               std::string* dbg)
{
    const char* cand[] = {
        "/dataset/image_0/data", "/dataset/image_0",
        "/dataset/images_0/data","/dataset/images_0",
        "/dataset/image","/dataset/images",
        "/dataset/reconstruction_rss",
        "image_0","images_0","image","images","reconstruction_rss"
    };

    for (auto p : cand) {
        try {
            if (!f.nameExists(p)) continue;
            H5::DataSet ds = f.openDataSet(p);
            H5T_class_t cls = ds.getDataType().getClass();
            int ty_ny=0, ty_nx=0; std::vector<float> tmp;

            if (cls == H5T_FLOAT && read_float_2D(ds, ty_ny, ty_nx, tmp, dbg)) {
                ny=ty_ny; nx=ty_nx; out.swap(tmp);
                dbg_line(dbg, "[ismrmrd] image path used: ", p);
                return true;
            }
            if (cls == H5T_COMPOUND && read_compound_2D_mag(ds, ty_ny, ty_nx, tmp, dbg)) {
                ny=ty_ny; nx=ty_nx; out.swap(tmp);
                dbg_line(dbg, "[ismrmrd] image path used: ", p);
                return true;
            }
            if (cls == H5T_FLOAT && read_tail2_2Dor3D_mag_slice0(ds, ty_ny, ty_nx, tmp, dbg)) {
                ny=ty_ny; nx=ty_nx; out.swap(tmp);
                dbg_line(dbg, "[ismrmrd] image path used (tail2): ", p);
                return true;
            }
        } catch (const H5::Exception&) { /* keep scanning */ }
    }
    return false;
}

// ---- k-space readers ----
static bool read_kspace_float_tail2(H5::DataSet& ds, H5::DataSpace& sp,
                                    mri::KSpace& ks, std::string* dbg) {
    int rank = sp.getSimpleExtentNdims();
    std::vector<hsize_t> dims(rank);
    sp.getSimpleExtentDims(dims.data());

    auto read_one = [&](hsize_t C, hsize_t ny, hsize_t nx,
                        const std::vector<hsize_t>& start) -> bool {
        ks.coils=(int)C; ks.ny=(int)ny; ks.nx=(int)nx;
        const size_t N=(size_t)C*ny*nx; ks.host.resize(N);
        std::vector<float> tmp(N*2);
        std::vector<hsize_t> count(rank); for (int i=0;i<rank;++i) count[i]=dims[i];
        H5::DataSpace fs = ds.getSpace();
        fs.selectHyperslab(H5S_SELECT_SET, count.data(), start.data());
        H5::DataSpace ms(rank, count.data());
        ds.read(tmp.data(), H5::PredType::NATIVE_FLOAT, ms, fs);
        for (size_t i=0;i<N;++i) ks.host[i] = std::complex<float>(tmp[2*i+0], tmp[2*i+1]);
        dbg_line(dbg, "[ismrmrd][k] tail-2: C=", ks.coils, " ny=", ks.ny, " nx=", ks.nx);
        return true;
    };

    if (rank==4 && dims[3]==2) return read_one(dims[0], dims[1], dims[2], {0,0,0,0});
    if (rank==5 && dims[4]==2) return read_one(dims[1], dims[2], dims[3], {0,0,0,0,0});
    return false;
}

static bool read_kspace_compound(H5::DataSet& ds, H5::DataSpace& sp,
                                 mri::KSpace& ks, std::string* dbg) {
    int rank = sp.getSimpleExtentNdims();
    std::vector<hsize_t> dims(rank);
    sp.getSimpleExtentDims(dims.data());

    H5::DataType dt = ds.getDataType();
    if (dt.getClass() != H5T_COMPOUND) return false;
    H5::CompType dsType(dt.getId());
    if (dsType.getNmembers() < 2) return false;

    H5::CompType memType(sizeof(float)*2);
    memType.insertMember(dsType.getMemberName(0), 0,             H5::PredType::NATIVE_FLOAT);
    memType.insertMember(dsType.getMemberName(1), sizeof(float), H5::PredType::NATIVE_FLOAT);

    auto read_full3 = [&](hsize_t C,hsize_t ny,hsize_t nx){
        ks.coils=(int)C; ks.ny=(int)ny; ks.nx=(int)nx;
        const size_t N=(size_t)C*ny*nx; ks.host.resize(N);
        std::vector<float> tmp(N*2);
        H5::DataSpace fs = ds.getSpace();
        std::vector<hsize_t> count(rank); for (int i=0;i<rank;++i) count[i]=dims[i];
        H5::DataSpace ms(rank, count.data());
        ds.read(tmp.data(), memType, ms, fs);
        for (size_t i=0;i<N;++i) ks.host[i]=std::complex<float>(tmp[2*i], tmp[2*i+1]);
        dbg_line(dbg, "[ismrmrd][k] compound: C=", ks.coils, " ny=", ks.ny, " nx=", ks.nx);
        return true;
    };
    auto read_slice4 = [&](hsize_t S,hsize_t C,hsize_t ny,hsize_t nx){
        if (S==0) return false;
        ks.coils=(int)C; ks.ny=(int)ny; ks.nx=(int)nx;
        const size_t N=(size_t)C*ny*nx; ks.host.resize(N);
        std::vector<float> tmp(N*2);
        H5::DataSpace fs = ds.getSpace();
        hsize_t start[4]={0,0,0,0}, count[4]={1,C,ny,nx};
        fs.selectHyperslab(H5S_SELECT_SET, count, start);
        H5::DataSpace ms(4, count);
        ds.read(tmp.data(), memType, ms, fs);
        for (size_t i=0;i<N;++i) ks.host[i]=std::complex<float>(tmp[2*i], tmp[2*i+1]);
        dbg_line(dbg, "[ismrmrd][k] compound slice0: C=", ks.coils, " ny=", ks.ny, " nx=", ks.nx);
        return true;
    };

    if (rank==3) return read_full3(dims[0], dims[1], dims[2]);
    if (rank==4) return read_slice4(dims[0], dims[1], dims[2], dims[3]);
    return false;
}

static bool try_read_any_kspace(H5::H5File& f, mri::KSpace& ks, std::string* dbg) {
    const char* cand[] = { "/dataset/kspace", "kspace" };
    for (auto p : cand) {
        try {
            if (!f.nameExists(p)) continue;
            H5::DataSet ds = f.openDataSet(p);
            H5::DataSpace sp = ds.getSpace();
            H5T_class_t cls = ds.getDataType().getClass();
            bool ok = false;
            if (cls == H5T_FLOAT)        ok = read_kspace_float_tail2(ds, sp, ks, dbg);
            else if (cls == H5T_COMPOUND) ok = read_kspace_compound(ds, sp, ks, dbg);
            if (ok) { dbg_line(dbg, "[ismrmrd] kspace path used: ", p); return true; }
        } catch (const H5::Exception&) { /* keep scanning */ }
    }
    return false;
}

// REQUIRE: at top of this file, add
//   #ifdef _OPENMP
//   #include <omp.h>
//   #endif

static bool assemble_from_acquisitions(const std::string& path,
                                       const std::string& xml,
                                       mri::KSpace& ks,
                                       std::string* dbg)
{
    dbg_line(dbg, "[ismrmrd][acq] attempting Cartesian assembly from /dataset/acquisitions.");

    // ---- 0) Header hints + hard guard on trajectory ---------------------------------
    int nx_hdr=0, ny_hdr=0, ymin_hdr=0, ycenter_hdr=0, ymax_hdr=-1;
    std::string traj = "unknown";
    parse_header_dims(xml, nx_hdr, ny_hdr, ymin_hdr, ycenter_hdr, ymax_hdr, traj, dbg);
    if (!traj.empty() && traj != "cartesian") {
        dbg_line(dbg, "[ismrmrd][guard] Non-cartesian trajectory '", traj, "' -> skipping acquisition assembly.");
        return false;
    }

    // ---- 1) Open dataset / probe acquisition count -----------------------------------
    ISMRMRD::Dataset ds(path.c_str(), "/dataset", false);
    const uint32_t nAcq = ds.getNumberOfAcquisitions();
    if (nAcq == 0) {
        dbg_line(dbg, "[ismrmrd][acq][ERR] no acquisitions");
        return false;
    }

#ifdef _OPENMP
    dbg_line(dbg, "[OMP][assemble] omp_get_max_threads()=", omp_get_max_threads());
#else
    dbg_line(dbg, "[OMP][assemble] OpenMP disabled at compile time");
#endif

    // ---- 2) PASS 0: scan index ranges after filtering obvious non-imaging flags ------
    struct Range { int mn=INT_MAX, mx=INT_MIN; void see(int v){ mn=std::min(mn,v); mx=std::max(mx,v);} };
    Range r_slice, r_set, r_phase, r_contrast, r_segment, r_rep, r_avg, r_ky;

    uint64_t skipped_noise=0, skipped_nav=0, skipped_calib=0;
    for (uint32_t i=0; i<nAcq; ++i) {
        ISMRMRD::Acquisition a; ds.readAcquisition(i, a);
        if (a.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NOISE_MEASUREMENT))             { ++skipped_noise; continue; }
        if (a.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NAVIGATION_DATA))               { ++skipped_nav;   continue; }
        if (a.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION) ||
            a.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING)) { ++skipped_calib; continue; }

        const auto& h = a.getHead();
        r_slice.see((int)h.idx.slice);
        r_set.see((int)h.idx.set);
        r_phase.see((int)h.idx.phase);
        r_contrast.see((int)h.idx.contrast);
        r_segment.see((int)h.idx.segment);
        r_rep.see((int)h.idx.repetition);
        r_avg.see((int)h.idx.average);
        r_ky.see((int)h.idx.kspace_encode_step_1);
    }

    auto choose = [](const Range& r){ return r.mn==INT_MAX ? 0 : r.mn; };
    const int keep_slice    = choose(r_slice);
    const int keep_set      = choose(r_set);
    const int keep_phase    = choose(r_phase);
    const int keep_contrast = choose(r_contrast);
    const int keep_segment  = choose(r_segment);
    const int keep_rep      = 0; // pick earliest if present (consistent with your code)
    const int keep_avg      = 0;

    dbg_line(dbg, "[ismrmrd][acq] index ranges:",
             " slice[", r_slice.mn, ",", r_slice.mx, "]",
             " set[",   r_set.mn,   ",", r_set.mx,   "]",
             " phase[", r_phase.mn, ",", r_phase.mx, "]",
             " contrast[", r_contrast.mn, ",", r_contrast.mx, "]",
             " segment[",  r_segment.mn,  ",", r_segment.mx,  "]",
             " repetition[", r_rep.mn, ",", r_rep.mx, "]",
             " average[",   r_avg.mn, ",", r_avg.mx, "]");
    dbg_line(dbg, "[ismrmrd][acq] selected subspace: slice=", keep_slice,
             " set=", keep_set, " phase=", keep_phase, " contrast=", keep_contrast,
             " segment=", keep_segment, " repetition=", keep_rep, " average=", keep_avg);

    if (skipped_noise || skipped_nav || skipped_calib) {
        dbg_line(dbg, "[ismrmrd][acq] skipped at scan{noise=", skipped_noise,
                 ", nav=", skipped_nav, ", calib=", skipped_calib, "}");
    }

    // ---- 3) PASS 1: decide dims from chosen subspace ---------------------------------
    int ky_min_obs = INT_MAX, ky_max_obs = INT_MIN, nxmax = 0, Cmax = 0;
    uint64_t accepted = 0;

    for (uint32_t i=0; i<nAcq; ++i) {
        ISMRMRD::Acquisition a; ds.readAcquisition(i, a);
        if (a.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NOISE_MEASUREMENT)) continue;
        if (a.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NAVIGATION_DATA))   continue;
        if (a.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION) ||
            a.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING)) continue;

        const auto& h = a.getHead();
        if ((int)h.idx.slice     != keep_slice)    continue;
        if ((int)h.idx.set       != keep_set)      continue;
        if ((int)h.idx.phase     != keep_phase)    continue;
        if ((int)h.idx.contrast  != keep_contrast) continue;
        if ((int)h.idx.segment   != keep_segment)  continue;
        if ((int)h.idx.repetition!= keep_rep)      continue;
        if ((int)h.idx.average   != keep_avg)      continue;

        ++accepted;
        ky_min_obs = std::min<int>(ky_min_obs, (int)h.idx.kspace_encode_step_1);
        ky_max_obs = std::max<int>(ky_max_obs, (int)h.idx.kspace_encode_step_1);
        nxmax      = std::max<int>(nxmax, a.number_of_samples());
        Cmax       = std::max<int>(Cmax, a.active_channels());
    }
    if (accepted == 0) {
        dbg_line(dbg, "[ismrmrd][acq][ERR] no acquisitions match selected subspace");
        return false;
    }

    // From header if consistent; otherwise from observed
    int ymin = (ymax_hdr >= ymin_hdr) ? ymin_hdr : ky_min_obs;
    int ymax = (ymax_hdr >= ymin_hdr) ? ymax_hdr : ky_max_obs;
    if (ny_hdr > 0 && ymax >= ny_hdr) {
        dbg_line(dbg, "[ismrmrd][acq][FIX] ymax(", ymax, ") >= ny_hdr(", ny_hdr, ") -> clamp to ", ny_hdr-1);
        ymax = ny_hdr - 1;
    }

    const int ny = std::max(0, ymax - ymin + 1);
    const int nx = (nx_hdr > 0) ? std::min(nx_hdr, nxmax) : nxmax;
    const int C  = std::max(1, Cmax);

    if (ny <= 0 || nx <= 0) {
        dbg_line(dbg, "[ismrmrd][acq][ERR] computed dims invalid: C=", C, " ny=", ny, " nx=", nx);
        return false;
    }

    ks.coils = C; ks.ny = ny; ks.nx = nx;
    ks.host.assign((size_t)C * ny * nx, std::complex<float>(0.f, 0.f));

    std::vector<uint8_t> written((size_t)C * ny, 0);  // guard for duplicates: one bit per (ky, coil)
    uint64_t filled_pairs = 0, skipped_dup = 0, clipped_samples = 0, padded_samples = 0;

    dbg_line(dbg, "[ismrmrd][acq] dims decided: C=", C, " ny=", ny, " nx=", nx,
             " (ymin=", ymin, " .. ymax=", ymax, ")");

    // ---- 4) PASS 2: fill k-space buffer (OpenMP on per-sample copy) ------------------
    for (uint32_t i=0; i<nAcq; ++i) {
        ISMRMRD::Acquisition a; ds.readAcquisition(i, a);
        // Same flag filters
        if (a.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NOISE_MEASUREMENT)) continue;
        if (a.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NAVIGATION_DATA))   continue;
        if (a.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION) ||
            a.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING)) continue;

        const auto& h = a.getHead();
        if ((int)h.idx.slice     != keep_slice)    continue;
        if ((int)h.idx.set       != keep_set)      continue;
        if ((int)h.idx.phase     != keep_phase)    continue;
        if ((int)h.idx.contrast  != keep_contrast) continue;
        if ((int)h.idx.segment   != keep_segment)  continue;
        if ((int)h.idx.repetition!= keep_rep)      continue;
        if ((int)h.idx.average   != keep_avg)      continue;

        const int ky = (int)h.idx.kspace_encode_step_1;
        if (ky < ymin || ky > ymax) continue;
        const int y = ky - ymin;

        int ns = a.number_of_samples();
        const int ch = std::min<int>(a.active_channels(), C);
        if (ns > nx) { clipped_samples += (uint64_t)(ns - nx) * ch; ns = nx; }

        for (int c = 0; c < ch; ++c) {
            const size_t pair = (size_t)y * C + (size_t)c;
            if (written[pair]) { ++skipped_dup; continue; } // first-seen wins

            const size_t base = (size_t)c * ny * nx + (size_t)y * nx;

#ifdef _OPENMP
// Parallelize the per-sample copy along readout (fast, independent)
#pragma omp parallel for if(ns > 64) schedule(static)
#endif
            for (int s = 0; s < ns; ++s) {
                ks.host[base + s] = a.data(s, c); // complex<float> copy
            }

            if (ns < nx) padded_samples += (uint64_t)(nx - ns);

            written[pair] = 1;
            ++filled_pairs;
        }
    }

    const size_t missing_pairs = (size_t)C * ny - filled_pairs;
    dbg_line(dbg, "[ismrmrd][acq] wrote unique (ky,coil) pairs = ", filled_pairs, " / ", (size_t)C*ny);
    dbg_line(dbg, "[ismrmrd][acq] skipped duplicates=", skipped_dup,
             " missing_pairs=", missing_pairs,
             " clipped_samples=", clipped_samples,
             " padded_samples=", padded_samples);
    if (missing_pairs > 0) {
        dbg_line(dbg, "[ismrmrd][acq][WARN] some (ky,coil) lines missing; zeros remain in those rows.");
    }

    return (filled_pairs > 0);
}

} // anonymous namespace

// -----------------------------------------------------------------------------
// public API
// -----------------------------------------------------------------------------
namespace mri {

bool load_ismrmrd_any(const std::string& path,
                      KSpace& ks,
                      std::vector<float>* preRecon,
                      int* preNy,
                      int* preNx,
                      std::string* dbg)
{
    try {
        suppress_hdf5_error_stack_once();
        dbg_line(dbg, "[ismrmrd] open: ", path);
        H5::H5File f(path, H5F_ACC_RDONLY);

        // Read XML once and decide trajectory (guard for non-Cartesian)
        std::string xml = read_xml_text(f);
        std::string traj = "unknown";
        if (!xml.empty()) {
            int nx_hdr=0, ny_hdr=0, ymin=0, ycenter=0, ymax=-1;
            parse_header_dims(xml, nx_hdr, ny_hdr, ymin, ycenter, ymax, traj, dbg);
        } else {
            dbg_line(dbg, "[ismrmrd] no /dataset/xml (trajectory unknown)");
        }

        if (!traj.empty() && traj != "cartesian") {
            dbg_line(dbg, "[ismrmrd][guard] Non-cartesian trajectory '", traj,
                     "' detected -> will NOT process k-space.");
            // Try a pre-reconstructed image if present
            int imy=0, imx=0; std::vector<float> img;
            if (try_read_any_image(f, img, imy, imx, dbg)) {
                if (preRecon) preRecon->swap(img);
                if (preNy) *preNy = imy;
                if (preNx) *preNx = imx;
                ks.coils = 1; ks.ny = imy; ks.nx = imx;
                dbg_line(dbg, "[ismrmrd][guard] Displaying embedded image only (", imy, "x", imx, ").");
                return true;
            }
            dbg_line(dbg, "[ismrmrd][guard] No embedded image found; aborting load.");
            return false;
        }

        // Happy path (Cartesian)
        try_read_ismrmrd_xml(f, dbg);

        bool gotImg=false, gotK=false;
        int imy=0, imx=0; std::vector<float> img;

        gotImg = try_read_any_image(f, img, imy, imx, dbg);
        gotK   = try_read_any_kspace(f, ks, dbg);

        if (!gotImg && !gotK) {
            if (assemble_from_acquisitions(path, xml, ks, dbg)) {
                gotK = true;
                dbg_line(dbg, "[ismrmrd][acq] k-space assembled from acquisitions.");
            } else {
                dbg_line(dbg, "[ismrmrd][acq] assembly failed.");
            }
        }

        if (gotImg && preRecon) {
            preRecon->swap(img);
            if (preNy) *preNy = imy;
            if (preNx) *preNx = imx;
        }
        if (gotK) return true;
        if (gotImg) { ks.coils=1; ks.ny=imy; ks.nx=imx; return true; }

        dbg_line(dbg, "[ismrmrd][ERR] no recognized image or k-space datasets");
        return false;
    } catch (const H5::Exception& e) {
        dbg_line(dbg, "[ismrmrd][ERR] HDF5: ", e.getDetailMsg());
        return false;
    } catch (const std::exception& e) {
        dbg_line(dbg, "[ismrmrd][ERR] ", e.what());
        return false;
    }
}

} // namespace mri
