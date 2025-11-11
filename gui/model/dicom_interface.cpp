#include "dicom_interface.hpp"
#include <QCoreApplication>
#include <QDir>

// DicomDll::load — resolves all required exports and prints loud diagnostics.
bool DicomDll::load(const QString& explicitPath)
{
    if (loaded) return true;

    QString candidate = explicitPath;
    if (candidate.isEmpty()) {
        const QString appDir = QCoreApplication::applicationDirPath();
        QStringList attempts;
#ifdef _WIN32
        attempts << (appDir + "/dicom_io_libd")   // Qt appends .dll automatically
                 << "dicom_io_libd"
                 << (QDir::currentPath() + "/dicom_io_libd");
#else
        attempts << (appDir + "/libdicom_io_libd.so")
                 << "libdicom_io_libd.so"
                 << (QDir::currentPath() + "/libdicom_io_libd.so");
#endif
        for (const auto& a : attempts) {
            std::cerr << "[DICOM][DLL][DBG] Trying to load: " << a.toStdString() << "\n";
            lib.setFileName(a);
            if (lib.load()) { candidate = a; break; }
        }
    } else {
        std::cerr << "[DICOM][DLL][DBG] Trying to load explicit: " << candidate.toStdString() << "\n";
        lib.setFileName(candidate);
        lib.load();
    }

    if (!lib.isLoaded()) {
        std::cerr << "[DICOM][DLL][ERR] Could not load dicom_io_libd ("
                  << candidate.toStdString() << ") : "
                  << lib.errorString().toStdString() << "\n";
        return false;
    }

    // Required symbols
    p_probe     = reinterpret_cast<PFN_probe>(lib.resolve("dicom_probe"));
    p_read      = reinterpret_cast<PFN_read >(lib.resolve("dicom_read_gray8"));
    p_free      = reinterpret_cast<PFN_free >(lib.resolve("dicom_free"));
    p_write     = reinterpret_cast<PFN_write>(lib.resolve("dicom_write_sc_gray8"));
    // NEW: extra helpers so controller stays clean
    p_count     = reinterpret_cast<PFN_count   >(lib.resolve("dicom_count_frames"));
    p_info      = reinterpret_cast<PFN_info    >(lib.resolve("dicom_info"));
    p_read_all  = reinterpret_cast<PFN_read_all>(lib.resolve("dicom_read_all_gray8"));

    const bool ok =
        p_probe && p_read && p_free && p_write &&
        p_count && p_info && p_read_all;

    if (!ok) {
        std::cerr << "[DICOM][DLL][ERR] Missing exports:"
                  << " probe="    << (p_probe   ? "OK":"NULL")
                  << " read="     << (p_read    ? "OK":"NULL")
                  << " free="     << (p_free    ? "OK":"NULL")
                  << " write="    << (p_write   ? "OK":"NULL")
                  << " count="    << (p_count   ? "OK":"NULL")
                  << " info="     << (p_info    ? "OK":"NULL")
                  << " read_all=" << (p_read_all? "OK":"NULL")
                  << "\n";
        lib.unload();
        return false;
    }

    loaded = true;
    std::cerr << "[DICOM][DLL][DBG] Loaded dicom_io_libd from: "
              << lib.fileName().toStdString() << "\n";
    return true;
}
