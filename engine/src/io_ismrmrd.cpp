#include "io_ismrmrd.hpp"
#include "common.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <complex>
#include <functional>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <mutex>

// Feature gates (override from your build system if needed)
#ifndef BENCHMARK
#define BENCHMARK 0
#endif

#ifndef USE_SLICE_CACHE
#define USE_SLICE_CACHE 1
#endif

// 1 = parallelize over acquisitions (ky rows), 0 = over coils
#ifndef PAR_KY
#define PAR_KY 1
#endif

#ifndef ENGINE_HAS_ISMRMRD
#define ENGINE_HAS_ISMRMRD 1
#endif

// Debug verbosity for per-line placement (heavy)
#ifndef DBG_IO_LINES
#define DBG_IO_LINES 0
#endif

// -----------------------------------------------------------------------------
// If ISMRMRD is disabled at build time, provide a stub that logs + fails
// -----------------------------------------------------------------------------
#if !ENGINE_HAS_ISMRMRD

bool load_ismrmrd_slice(const std::string& path, int slice_idx, int step2, KsGrid& ks) {
    (void)path; (void)slice_idx; (void)step2; (void)ks;
    dbg_head("IO"); std::cerr << "ISMRMRD disabled at build; cannot load slice.\n";
    return false;
}

#else // ENGINE_HAS_ISMRMRD

// External deps (header-only PugiXML + ISMRMRD C++ API)
#include <pugixml.hpp>
#include <ismrmrd/ismrmrd.h>
#include <ismrmrd/dataset.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// =================== Benchmark helpers ===================
#if BENCHMARK
using bench_clock = std::chrono::high_resolution_clock;
#define TIC(name) auto name##_tic = bench_clock::now()
#define TOC(name, label)                                                      \
    do {                                                                      \
        auto name##_toc = bench_clock::now();                                 \
        double name##_ms = std::chrono::duration<double, std::milli>(         \
                             name##_toc - name##_tic).count();                 \
        dbg_head("BENCH"); std::cerr << label << " = " << name##_ms << " ms\n"; \
    } while(0)
#else
#define TIC(name)       do{}while(0)
#define TOC(name,label) do{}while(0)
#endif

// =================== Info logging toggle ===================
#if BENCHMARK
#define IO_INFO 0
#else
#define IO_INFO 1
#endif

#if IO_INFO
#define IO_LOG(tag, msg) do { dbg_head(tag); std::cerr << msg; } while(0)
#else
#define IO_LOG(tag, msg) do{}while(0)
#endif

// ======================================================================================
// Helpers
// ======================================================================================
namespace {

    struct HeaderSizes {
        int encX = 0, encY = 0, recX = 0;
        bool have_step2 = false;
        int step2_center = 0;
    };

    static HeaderSizes parse_sizes_from_header(const pugi::xml_node& enc) {
        HeaderSizes s;
        s.encX = xml_int(enc, "encodedSpace/matrixSize/x", 0);
        s.encY = xml_int(enc, "encodedSpace/matrixSize/y", 0);
        s.recX = xml_int(enc, "reconSpace/matrixSize/x", s.encX);

        auto lim_step2 = enc.child("encodingLimits").child("kspace_encoding_step_2");
        s.have_step2 = (bool)lim_step2 && lim_step2.child("maximum");
        s.step2_center = lim_step2.child("center").text().as_int(0);

        IO_LOG("IO", "Header sizes: enc=(" << s.encX << "," << s.encY
            << ") recX=" << s.recX << " have_step2=" << s.have_step2
            << " center2=" << s.step2_center << "\n");
        return s;
    }

    struct Step2Policy { bool use = false; int value = 0; };
    static Step2Policy decide_step2(int step2_arg, const HeaderSizes& hs) {
        Step2Policy p;
        if (!hs.have_step2) {
            p.use = false; p.value = 0;
            IO_LOG("IO", "Step2: not present in header (no filter)\n");
            return p;
        }
        p.use = true;
        p.value = (step2_arg >= 0) ? step2_arg : hs.step2_center;
        IO_LOG("IO", "Step2: using value=" << p.value
            << (step2_arg >= 0 ? " (from arg)" : " (from center)") << "\n");
        return p;
    }

    using AcceptFn = std::function<bool(const ISMRMRD::AcquisitionHeader&)>;

    static AcceptFn make_accept_predicate(int slice_idx, const Step2Policy& step2) {
        return [=](const ISMRMRD::AcquisitionHeader& h) -> bool {
            if (h.flags & ISMRMRD::ISMRMRD_ACQ_IS_NOISE_MEASUREMENT) return false;
            if (h.flags & ISMRMRD::ISMRMRD_ACQ_IS_NAVIGATION_DATA)   return false;
            if ((int)h.idx.slice != slice_idx)                        return false;
            if (step2.use && (int)h.idx.kspace_encode_step_2 != step2.value) return false;
            if (h.idx.repetition != 0)                                return false;
            if (h.idx.set != 0 || h.idx.average != 0 || h.idx.contrast != 0
                || h.idx.phase != 0 || h.idx.segment != 0)            return false;
            return true;
            };
    }

    // ---- first pass for a single slice (legacy path) ----
    struct ScanDims {
        int C = 0;
        int max_ky = -1;
        int nsamp_max = 0;
        bool saw_any = false;
    };

    static ScanDims first_pass_scan(ISMRMRD::Dataset& d, const AcceptFn& accept) {
        ScanDims r;
        const uint64_t n = d.getNumberOfAcquisitions();
        ISMRMRD::Acquisition acq;

        for (uint64_t i = 0; i < n; ++i) {
            d.readAcquisition(i, acq);
            const auto& h = acq.getHead();
            if (!accept(h)) continue;

            r.saw_any = true;
            r.C = std::max(r.C, (int)h.active_channels);
            r.max_ky = std::max(r.max_ky, (int)h.idx.kspace_encode_step_1);

            const int nsamp = (int)h.number_of_samples;
            const int pre = (int)h.discard_pre;
            const int post = (int)h.discard_post;
            const int kept = std::max(0, nsamp - pre - post);
            r.nsamp_max = std::max(r.nsamp_max, kept);
        }

        IO_LOG("IO", "First pass: saw=" << r.saw_any
            << " C=" << r.C << " max_ky=" << r.max_ky
            << " nsamp_max=" << r.nsamp_max << "\n");
        return r;
    }

    static void alloc_grid(KsGrid& ks, const ScanDims& dims, int recX, int encY) {
        const int nx = std::max(recX, dims.nsamp_max);
        const int ny = std::max(dims.max_ky + 1, encY);  // simple and safe
        ks.coils = dims.C;
        ks.nx = nx;
        ks.ny = ny;
        ks.host.assign((size_t)ks.coils * ks.ny * ks.nx, std::complex<float>(0, 0));
        IO_LOG("IO", "Alloc grid: C=" << ks.coils
            << " ny=" << ks.ny << " nx=" << ks.nx
            << " total=" << ks.host.size() << " complex\n");
    }

    struct LineMap {
        int start_out = 0;
        int eff_off = 0;
        int eff_len = 0;
        int center_in = 0;
        int nx_half = 0;
    };

    static inline int clampv(int v, int lo, int hi) {
        if (v < lo) return lo;
        if (v > hi) return hi;
        return v;
    }

    static LineMap plan_line_map(int nx, int nsamp, int pre, int post, int center_sample) {
        LineMap m;
        const int kept = std::max(0, nsamp - pre - post);
        if (kept <= 0) return m;

        int center_raw = center_sample;
        if (center_raw <= 0 || center_raw >= nsamp) center_raw = pre + kept / 2;

        m.center_in = clampv(center_raw - pre, 0, std::max(0, kept - 1));
        m.nx_half = nx / 2;
        m.start_out = m.nx_half - m.center_in;
        m.eff_off = 0;
        m.eff_len = kept;

        if (m.start_out < 0) {
            int drop = -m.start_out;
            m.eff_off += drop;
            m.eff_len -= drop;
            m.start_out = 0;
        }
        if (m.start_out + m.eff_len > nx) {
            m.eff_len -= (m.start_out + m.eff_len - nx);
        }

        IO_LOG("Map", "center_in=" << m.center_in
            << " nx_half=" << m.nx_half
            << " start_out=" << m.start_out
            << " eff_off=" << m.eff_off
            << " eff_len=" << m.eff_len << "\n");
        return m;
    }

    // =================== Prefetch ===================
    struct PrefetchedAcq {
        int ky = 0;
        int nsamp = 0;
        int pre = 0;
        int post = 0;
        int center_sample = 0;
        int active_channels = 0;
        std::vector<std::complex<float>> data; // [active_channels, nsamp]
    };

    static void prefetch_from_ids(ISMRMRD::Dataset& d,
        const std::vector<uint64_t>& ids,
        std::vector<PrefetchedAcq>& out)
    {
        out.clear();
        out.reserve(ids.size());
        ISMRMRD::Acquisition acq;
        for (uint64_t id : ids) {
            d.readAcquisition(id, acq);
            const auto& h = acq.getHead();

            PrefetchedAcq pa;
            pa.ky = (int)h.idx.kspace_encode_step_1;
            pa.nsamp = (int)h.number_of_samples;
            pa.pre = (int)h.discard_pre;
            pa.post = (int)h.discard_post;
            pa.center_sample = (int)h.center_sample;
            pa.active_channels = (int)h.active_channels;

            const auto* src = reinterpret_cast<const std::complex<float>*>(acq.getDataPtr());
            pa.data.resize((size_t)pa.active_channels * (size_t)pa.nsamp);
            std::memcpy(pa.data.data(), src, pa.data.size() * sizeof(std::complex<float>));

            out.push_back(std::move(pa));
        }
    }

    static void prefetch_from_dataset(ISMRMRD::Dataset& d,
        const AcceptFn& accept,
        std::vector<PrefetchedAcq>& out)
    {
        out.clear();
        ISMRMRD::Acquisition acq;
        const uint64_t n = d.getNumberOfAcquisitions();
        out.reserve((size_t)std::min<uint64_t>(n, 2048));

        for (uint64_t i = 0; i < n; ++i) {
            d.readAcquisition(i, acq);
            const auto& h = acq.getHead();
            if (!accept(h)) continue;

            PrefetchedAcq pa;
            pa.ky = (int)h.idx.kspace_encode_step_1;
            pa.nsamp = (int)h.number_of_samples;
            pa.pre = (int)h.discard_pre;
            pa.post = (int)h.discard_post;
            pa.center_sample = (int)h.center_sample;
            pa.active_channels = (int)h.active_channels;

            const auto* src = reinterpret_cast<const std::complex<float>*>(acq.getDataPtr());
            pa.data.resize((size_t)pa.active_channels * (size_t)pa.nsamp);
            std::memcpy(pa.data.data(), src, pa.data.size() * sizeof(std::complex<float>));

            out.push_back(std::move(pa));
        }
    }

    // =================== memcpy-based copy ===================
    static void copy_prefetched_to_grid(const PrefetchedAcq& pa,
        const LineMap& m,
        int ky,
        int C, int ny, int nx,
        std::vector<std::complex<float>>& host,
        bool parallel_over_coils)
    {
        if (m.eff_len <= 0) return;

        const int maxC = std::min(pa.active_channels, C);
        const size_t plane_stride = (size_t)ny * (size_t)nx;
        const size_t ky_offset = (size_t)ky * (size_t)nx;
        const int src_base = pa.pre + m.eff_off;

        auto do_copy_for_coil = [&](int c) {
            const std::complex<float>* line = pa.data.data() + (size_t)c * pa.nsamp + src_base;
            std::complex<float>* dst = &host[(size_t)c * plane_stride + ky_offset] + m.start_out;
            std::memcpy(dst, line, (size_t)m.eff_len * sizeof(std::complex<float>));
            };

        if (parallel_over_coils) {
#pragma omp parallel for if(maxC>1) schedule(static)
            for (int c = 0; c < maxC; ++c) do_copy_for_coil(c);
        }
        else {
            for (int c = 0; c < maxC; ++c) do_copy_for_coil(c);
        }

#if DBG_IO_LINES
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        std::cerr << "[DBG][IO] ky=" << ky
            << " start=" << m.start_out << " len=" << m.eff_len
            << " center_in=" << m.center_in
            << " tid=" << tid << "\n";
#endif
    }

    // =================== Slice cache (thread-safe) ===================
#if USE_SLICE_CACHE
    struct SliceEntry {
        int C = 0;
        int max_ky = -1;
        int nsamp_max = 0;
        std::vector<uint64_t> acq_ids; // indices into dataset
    };

    struct FileSliceCache {
        // key params baked into the cache:
        std::string path;
        bool step2_use = false;
        int  step2_val = 0;
        int  recX = 0, encY = 0;  // from header
        int  min_slice = 0, max_slice = 0;

        std::vector<SliceEntry> slices;
        bool ready = false;
    };

    static std::unordered_map<std::string, FileSliceCache> g_slice_cache;
    static std::mutex g_cache_mtx;

    static std::string make_cache_key(const std::string& path, const Step2Policy& step) {
        return path + "#" + (step.use ? "1" : "0") + ":" + std::to_string(step.value);
    }

    static FileSliceCache& get_or_build_cache(const std::string& path,
        ISMRMRD::Dataset& d,
        const pugi::xml_node& enc,
        const Step2Policy& step)
    {
        HeaderSizes hs = parse_sizes_from_header(enc);
        const std::string key = make_cache_key(path, step);

        {
            std::scoped_lock<std::mutex> lock(g_cache_mtx);
            auto it = g_slice_cache.find(key);
            if (it != g_slice_cache.end() && it->second.ready) {
                return it->second;
            }
        }

        // Build cache (outside the lock — dataset reads are expensive)
        TIC(preindex);
        FileSliceCache fresh;
        fresh.path = path;
        fresh.step2_use = step.use;
        fresh.step2_val = step.value;
        fresh.recX = hs.recX;
        fresh.encY = hs.encY;

        // slice bounds from header
        auto lim_slice = enc.child("encodingLimits").child("slice");
        fresh.min_slice = lim_slice.child("minimum").text().as_int(0);
        fresh.max_slice = lim_slice.child("maximum").text().as_int(0);
        const int S = fresh.max_slice - fresh.min_slice + 1;
        fresh.slices.assign(S, SliceEntry{});

        // Accept predicate without slice filter (we bucket by slice ourselves)
        AcceptFn accept_no_slice = [&](const ISMRMRD::AcquisitionHeader& h)->bool {
            if (h.flags & ISMRMRD::ISMRMRD_ACQ_IS_NOISE_MEASUREMENT) return false;
            if (h.flags & ISMRMRD::ISMRMRD_ACQ_IS_NAVIGATION_DATA)   return false;
            if (step.use && (int)h.idx.kspace_encode_step_2 != step.value) return false;
            if (h.idx.repetition != 0) return false;
            if (h.idx.set != 0 || h.idx.average != 0 || h.idx.contrast != 0
                || h.idx.phase != 0 || h.idx.segment != 0) return false;
            return true;
            };

        const uint64_t n = d.getNumberOfAcquisitions();
        ISMRMRD::Acquisition acq;
        for (uint64_t i = 0; i < n; ++i) {
            d.readAcquisition(i, acq);
            const auto& h = acq.getHead();
            if (!accept_no_slice(h)) continue;

            const int s = (int)h.idx.slice;
            if (s < fresh.min_slice || s > fresh.max_slice) continue;
            SliceEntry& se = fresh.slices[s - fresh.min_slice];

            se.C = std::max(se.C, (int)h.active_channels);
            se.max_ky = std::max(se.max_ky, (int)h.idx.kspace_encode_step_1);

            const int nsamp = (int)h.number_of_samples;
            const int pre = (int)h.discard_pre;
            const int post = (int)h.discard_post;
            const int kept = std::max(0, nsamp - pre - post);
            se.nsamp_max = std::max(se.nsamp_max, kept);

            se.acq_ids.push_back(i);
        }

        fresh.ready = true;

        // Publish under lock
        {
            std::scoped_lock<std::mutex> lock(g_cache_mtx);
            g_slice_cache[key] = std::move(fresh);
        }
        TOC(preindex, "slice_cache_build_ms");

        // Return reference to the stored cache
        std::scoped_lock<std::mutex> lock2(g_cache_mtx);
        return g_slice_cache[key];
    }
#endif // USE_SLICE_CACHE

} // namespace (helpers)

// ======================================================================================
// Load ONE slice -> [C, ny, nx] k-space
// ======================================================================================
bool load_ismrmrd_slice(const std::string& path, int slice_idx, int step2, KsGrid& ks)
{
    try {
        const char* group = "dataset";
        ISMRMRD::Dataset d(path.c_str(), group, false);

        // (1) Header read + XML parse
        TIC(hdr);
        std::string xml; d.readHeader(xml);
        pugi::xml_document doc;
        if (!doc.load_string(xml.c_str())) {
            dbg_head("IO"); std::cerr << "XML parse failed; cannot size output\n";
            TOC(hdr, "hdr_xml_ms");
            return false;
        }
        auto H = doc.child("ismrmrdHeader");
        auto enc = H.child("encoding");
        TOC(hdr, "hdr_xml_ms");

        HeaderSizes sizes = parse_sizes_from_header(enc);
        Step2Policy step = decide_step2(step2, sizes);

#ifdef _OPENMP
        static bool printed = false;
        if (!printed) {
            printed = true;
            IO_LOG("OMP", "OpenMP enabled in io_ismrmrd.cpp, max_threads="
                << omp_get_max_threads() << "\n");
        }
#endif
        IO_LOG("IO", "Slice request: slice=" << slice_idx
            << (step.use ? " step2=" + std::to_string(step.value) : " (no step2)")
            << "  recX=" << sizes.recX << " encY=" << sizes.encY << "\n");

#if USE_SLICE_CACHE
        // Cached path (best performance)
        FileSliceCache& cache = get_or_build_cache(path, d, enc, step);

        const int sidx = slice_idx - cache.min_slice;
        if (sidx < 0 || sidx >= (int)cache.slices.size()) {
            dbg_head("IO"); std::cerr << "slice out of range\n";
            return false;
        }
        const SliceEntry& se = cache.slices[sidx];
        if (se.acq_ids.empty()) {
            dbg_head("IO"); std::cerr << "no acquisitions for slice\n";
            return false;
        }

        // Build dims from cached per-slice stats
        ScanDims dims;
        dims.C = se.C;
        dims.max_ky = se.max_ky;
        dims.nsamp_max = se.nsamp_max;
        dims.saw_any = true;

        // (3) Allocate
        TIC(alloc);
        alloc_grid(ks, dims, cache.recX, cache.encY);
        TOC(alloc, "alloc_grid_ms");

        // Prefetch slice acquisitions
        std::vector<PrefetchedAcq> pre;
        prefetch_from_ids(d, se.acq_ids, pre);

        // (4) Second pass copy
        TIC(pass2);
#if PAR_KY
        // Parallel over acquisitions (ky rows) — inner copy uses memcpy
#pragma omp parallel for schedule(static) if(pre.size() > 1)
        for (int i = 0; i < (int)pre.size(); ++i) {
            const PrefetchedAcq& pa = pre[i];
            const int ky = pa.ky;
            if (ky < 0 || ky >= ks.ny) continue;

            LineMap m = plan_line_map(ks.nx, pa.nsamp, pa.pre, pa.post, pa.center_sample);
            if (m.eff_len <= 0) continue;

            // Outer loop is parallel -> do NOT parallelize inside
            copy_prefetched_to_grid(pa, m, ky, ks.coils, ks.ny, ks.nx, ks.host, /*parallel_over_coils*/false);
        }
#else
        // Legacy: sequential over acquisitions; parallelize across coils inside
        for (const PrefetchedAcq& pa : pre) {
            const int ky = pa.ky;
            if (ky < 0 || ky >= ks.ny) continue;

            LineMap m = plan_line_map(ks.nx, pa.nsamp, pa.pre, pa.post, pa.center_sample);
            if (m.eff_len <= 0) continue;

            copy_prefetched_to_grid(pa, m, ky, ks.coils, ks.ny, ks.nx, ks.host, /*parallel_over_coils*/true);
        }
#endif
        TOC(pass2, "pass2_copy_ms");
        return true;

#else
        // Legacy path (per-slice scan without cache)
        // (2) First pass scan
        TIC(pass1);
        AcceptFn accept = make_accept_predicate(slice_idx, step);
        ScanDims dims = first_pass_scan(d, accept);
        TOC(pass1, "pass1_scan_ms");
        if (!dims.saw_any || dims.C <= 0 || dims.max_ky < 0) {
            dbg_head("IO"); std::cerr << "No data for slice=" << slice_idx
                << (step.use ? " step2=" + std::to_string(step.value) : "")
                << "\n";
            return false;
        }

        // (3) Allocate
        TIC(alloc);
        alloc_grid(ks, dims, sizes.recX, sizes.encY);
        TOC(alloc, "alloc_grid_ms");

        // Prefetch accepted acquisitions for this slice
        std::vector<PrefetchedAcq> pre;
        prefetch_from_dataset(d, accept, pre);

        // (4) Second pass copy
        TIC(pass2);
#if PAR_KY
#pragma omp parallel for schedule(static) if(pre.size() > 1)
        for (int i = 0; i < (int)pre.size(); ++i) {
            const PrefetchedAcq& pa = pre[i];
            const int ky = pa.ky;
            if (ky < 0 || ky >= ks.ny) continue;

            LineMap m = plan_line_map(ks.nx, pa.nsamp, pa.pre, pa.post, pa.center_sample);
            if (m.eff_len <= 0) continue;

            copy_prefetched_to_grid(pa, m, ky, ks.coils, ks.ny, ks.nx, ks.host, /*parallel_over_coils*/false);
        }
#else
        for (const PrefetchedAcq& pa : pre) {
            const int ky = pa.ky;
            if (ky < 0 || ky >= ks.ny) continue;

            LineMap m = plan_line_map(ks.nx, pa.nsamp, pa.pre, pa.post, pa.center_sample);
            if (m.eff_len <= 0) continue;

            copy_prefetched_to_grid(pa, m, ky, ks.coils, ks.ny, ks.nx, ks.host, /*parallel_over_coils*/true);
        }
#endif
        TOC(pass2, "pass2_copy_ms");
        return true;
#endif
    }
    catch (const std::exception& e) {
        dbg_head("IO"); std::cerr << "ERROR: " << e.what() << "\n";
        return false;
    }
    catch (...) {
        dbg_head("IO"); std::cerr << "Unknown error while loading slice\n";
        return false;
    }
}

#endif // ENGINE_HAS_ISMRMRD
