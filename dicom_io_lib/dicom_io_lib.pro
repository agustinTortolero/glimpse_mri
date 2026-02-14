# ================= dicom_io_lib.pro (Windows MSVC + Linux) ===================
QT       -= gui widgets
TEMPLATE  = lib
CONFIG   += c++17 dll warn_on
TARGET    = dicom_io_lib

# predictable per-config dirs
DESTDIR     = $$OUT_PWD
OBJECTS_DIR = $$OUT_PWD/obj
MOC_DIR     = $$OUT_PWD/moc
RCC_DIR     = $$OUT_PWD/rcc
UI_DIR      = $$OUT_PWD/ui

HEADERS += include/dicom_api.h
SOURCES += src/dicom_io_lib.cpp

DEFINES += DICOM_IO_LIB_BUILD
win32:msvc:DEFINES += _CRT_SECURE_NO_WARNINGS

# warnings
win32:msvc { QMAKE_CXXFLAGS += /W4 /permissive- /Zc:__cplusplus }
unix       { QMAKE_CXXFLAGS += -Wall -Wextra -Wpedantic
             CONFIG += hide_symbols }

# add "d" suffix for Debug on MSVC (dicom_io_libd.dll/.lib)
win32:msvc:CONFIG(debug, debug|release) { TARGET = $$join(TARGET,,,"d") }

# If someone selects a non-MSVC kit on Windows, stop immediately
win32:!msvc {
    message([ERR][DLL] This project is configured for **MSVC** on Windows.)
    error(Select an MSVC kit (e.g. Desktop Qt 6.x MSVC2022 64bit).)
}

# Optional: enable extra codecs if your DCMTK build has them (vcpkg feature flags)
isEmpty(CODECS_ENABLED) { CODECS_ENABLED = 0 }
defineReplace(dcmtk_codec_libs) {
    !equals(CODECS_ENABLED, 1): return()
    return(-ldcmjpeg -lcharls -ljpeg -lpng -ltiff -lz)
}

# ======================= Dependency: DCMTK ===================================

# ----------------------- Windows (MSVC + vcpkg) ------------------------------
win32:msvc {
    message([DBG][DLL] Platform = Windows (MSVC + vcpkg))

    # Root candidates (env first, then common locations)
    VCPKG_ROOT_ENV  = $$clean_path($$getenv(VCPKG_ROOT))
    LOCALAPPDATA    = $$clean_path($$getenv(LOCALAPPDATA))
    USERPROFILE     = $$clean_path($$getenv(USERPROFILE))

    VCPKG_ROOT_CANDIDATES =
    !isEmpty(VCPKG_ROOT_ENV): VCPKG_ROOT_CANDIDATES += $$VCPKG_ROOT_ENV
    VCPKG_ROOT_CANDIDATES += \
        C:/src/vcpkg \
        C:/vcpkg \
        $$LOCALAPPDATA/vcpkg \
        $$USERPROFILE/vcpkg \
        C:/Program\ Files/Microsoft\ Visual\ Studio/2022/Community/VC/vcpkg

    # Triplet (MSVC default)
    isEmpty(VCPKG_TRIPLET) { VCPKG_TRIPLET = x64-windows }

    message([DBG][DLL] Triplet         = $$VCPKG_TRIPLET)
    message([DBG][DLL] Roots to probe  = $$VCPKG_ROOT_CANDIDATES)

    MATCHED_ROOT =
    for(r, VCPKG_ROOT_CANDIDATES) {
        HDR = $$r/installed/$$VCPKG_TRIPLET/include/dcmtk/config/osconfig.h
        message([DBG][DLL] check: $$HDR)
        exists($$HDR) { MATCHED_ROOT = $$r }
    }

    isEmpty(MATCHED_ROOT) {
        message([ERR][DLL] dcmtk/config/osconfig.h not found for triplet: $$VCPKG_TRIPLET)
        message([ERR][DLL] Searched roots: $$VCPKG_ROOT_CANDIDATES)
        message([TIP][DLL] Install DCMTK: vcpkg install dcmtk:$$VCPKG_TRIPLET)
        message([TIP][DLL] Or pass: qmake "VCPKG_ROOT=C:/src/vcpkg" "VCPKG_TRIPLET=x64-windows")
        error(DCMTK headers missing -- cannot continue)
    }

    VCPKG_ROOT      = $$MATCHED_ROOT
    VCPKG_INSTALLED = $$VCPKG_ROOT/installed/$$VCPKG_TRIPLET
    DCMTK_INC       = $$VCPKG_INSTALLED/include
    CONFIG(debug, debug|release) { DCMTK_LIB = $$VCPKG_INSTALLED/debug/lib } else { DCMTK_LIB = $$VCPKG_INSTALLED/lib }

    message([DBG][DLL] Using VCPKG_ROOT = $$VCPKG_ROOT)
    message([DBG][DLL] DCMTK_INC       = $$DCMTK_INC)
    message([DBG][DLL] DCMTK_LIB       = $$DCMTK_LIB)

    INCLUDEPATH += $$quote($$DCMTK_INC)
    LIBS       += -L$$quote($$DCMTK_LIB)

    # Core DCMTK libs
    LIBS += -ldcmimgle -ldcmimage -ldcmdata -lofstd -loflog $$dcmtk_codec_libs()

    # Common Win system libs DCMTK may use
    LIBS += -lws2_32 -lwsock32 -liphlpapi -ladvapi32 -luser32
}

# ------------------------- Linux (Ubuntu/Raspbian) ---------------------------
unix {
    isEmpty(USE_VCPKG) { USE_VCPKG = 0 }

    equals(USE_VCPKG, 1) {
        message([DBG][DLL] Platform = Linux (vcpkg))
        VCPKG_ROOT = $$clean_path($$getenv(VCPKG_ROOT))
        isEmpty(VCPKG_ROOT) { VCPKG_ROOT = /usr/local/vcpkg }

        isEmpty(VCPKG_TRIPLET) {
            VCPKG_TRIPLET = $$system(uname -m)
            equals(VCPKG_TRIPLET, aarch64): VCPKG_TRIPLET = arm64-linux
            equals(VCPKG_TRIPLET, armv7l):  VCPKG_TRIPLET = arm-linux
            equals(VCPKG_TRIPLET, x86_64):  VCPKG_TRIPLET = x64-linux
        }

        VCPKG_INSTALLED = $$VCPKG_ROOT/installed/$$VCPKG_TRIPLET
        DCMTK_INC = $$VCPKG_INSTALLED/include
        CONFIG(debug, debug|release) { DCMTK_LIB = $$VCPKG_INSTALLED/debug/lib } else { DCMTK_LIB = $$VCPKG_INSTALLED/lib }

        !exists($$DCMTK_INC/dcmtk/config/osconfig.h) {
            message([ERR][DLL] DCMTK header NOT found: $$DCMTK_INC/dcmtk/config/osconfig.h)
            message([ERR][DLL] Fix: vcpkg install dcmtk:$$VCPKG_TRIPLET  (or use system pkg via pkg-config))
            error(DCMTK headers missing -- cannot continue)
        }

        INCLUDEPATH += $$quote($$DCMTK_INC)
        LIBS += -L$$quote($$DCMTK_LIB)
        LIBS += -ldcmimgle -ldcmimage -ldcmdata -lofstd -loflog $$dcmtk_codec_libs() -lpthread
        message([DBG][DLL] VCPKG_ROOT = $$VCPKG_ROOT)
        message([DBG][DLL] TRIPLET    = $$VCPKG_TRIPLET)
        message([DBG][DLL] INC/LIB    = $$DCMTK_INC | $$DCMTK_LIB)
    } else {
        # System packages (recommended):
        #   sudo apt-get install libdcmtk-dev pkg-config
        DCMTK_PC_OK = $$system(pkg-config --exists dcmtk && echo 1)
equals(DCMTK_PC_OK, 1) {
    CONFIG += link_pkgconfig
    PKGCONFIG += dcmtk
    message([DBG][DLL] Using pkg-config: dcmtk)
} else {
    message([WRN][DLL] dcmtk.pc not found. Falling back to manual include+libs.)
    INCLUDEPATH += /usr/include/dcmtk
    LIBS += -ldcmimgle -ldcmimage -ldcmdata -lofstd -loflog $$dcmtk_codec_libs() -lpthread -lz
}

        message([DBG][DLL] Platform = Linux (pkg-config: dcmtk))
        # If your distro’s dcmtk.pc doesn’t pull pthread/zlib, uncomment:
        # LIBS += -lpthread -lz
    }
}

# -------- final debug prints --------------------------------------------------
message([DBG][DLL] QMAKE_CXX   = $$QMAKE_CXX)
message([DBG][DLL] OUT_PWD     = $$OUT_PWD)
message([DBG][DLL] DESTDIR     = $$DESTDIR)
message([DBG][DLL] OBJ_DIR     = $$OBJECTS_DIR)
message([DBG][DLL] MOC_DIR     = $$MOC_DIR)
message([DBG][DLL] CONFIG      = $$CONFIG)
# ============================================================================
