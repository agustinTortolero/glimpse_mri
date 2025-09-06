QT += core
CONFIG += console c++17
CONFIG -= app_bundle
TEMPLATE = app

# ---------- vcpkg roots (forward slashes) ----------
VCPKG_ROOT = C:/src/vcpkg/installed/x64-windows
VCPKG_INC  = $$VCPKG_ROOT/include
VCPKG_LIBR = $$VCPKG_ROOT/lib
VCPKG_LIBD = $$VCPKG_ROOT/debug/lib
VCPKG_BINR = $$VCPKG_ROOT/bin
VCPKG_BIND = $$VCPKG_ROOT/debug/bin

INCLUDEPATH += $$VCPKG_INC

# helpful prints
message([DBG] VCPKG_INC  = $$VCPKG_INC)
message([DBG] VCPKG_LIBR = $$VCPKG_LIBR)
message([DBG] VCPKG_LIBD = $$VCPKG_LIBD)
message([DBG] VCPKG_BINR = $$VCPKG_BINR)
message([DBG] VCPKG_BIND = $$VCPKG_BIND)

# -------- RELEASE --------
win32:CONFIG(release, debug|release) {
    message([DBG] Build = RELEASE)
    LIBS += -L$$VCPKG_LIBR
    LIBS += -lhdf5 -lhdf5_cpp -lhdf5_hl -lhdf5_hl_cpp
    LIBS += -lzlib -laec

    # Copy runtime DLLs next to the Release exe (wildcards, quiet)
    DEST_DLL_DIR = $$OUT_PWD/release
    QMAKE_POST_LINK += cmd /c echo [DBG] Copy DLLs Release from \"$$VCPKG_BINR\" to \"$$DEST_DLL_DIR\" && \
        xcopy /Y /I /Q \"$$VCPKG_BINR\\hdf5*.dll\"  \"$$DEST_DLL_DIR\\\"  >NUL && \
        xcopy /Y /I /Q \"$$VCPKG_BINR\\zlib*.dll\"  \"$$DEST_DLL_DIR\\\"  >NUL && \
        xcopy /Y /I /Q \"$$VCPKG_BINR\\*aec*.dll\"  \"$$DEST_DLL_DIR\\\"  >NUL
}

# -------- DEBUG --------
win32:CONFIG(debug, debug|release) {
    message([DBG] Build = DEBUG)
    LIBS += -L$$VCPKG_LIBD
    # Your vcpkg debug libs have _D suffix
    LIBS += -lhdf5_D -lhdf5_cpp_D -lhdf5_hl_D -lhdf5_hl_cpp_D
    # zlibd exists; aec has no 'd' suffix (import lib is aec.lib)
    LIBS += -lzlibd -laec

    # Copy runtime DLLs next to the Debug exe (wildcards handle *_D.dll vs plain)
    DEST_DLL_DIR = $$OUT_PWD/debug
    QMAKE_POST_LINK += cmd /c echo [DBG] Copy DLLs Debug from \"$$VCPKG_BIND\" to \"$$DEST_DLL_DIR\" && \
        xcopy /Y /I /Q \"$$VCPKG_BIND\\hdf5*.dll\"  \"$$DEST_DLL_DIR\\\"  >NUL && \
        xcopy /Y /I /Q \"$$VCPKG_BIND\\zlib*.dll\"  \"$$DEST_DLL_DIR\\\"  >NUL && \
        xcopy /Y /I /Q \"$$VCPKG_BIND\\*aec*.dll\"  \"$$DEST_DLL_DIR\\\"  >NUL
}

# Optional: see where the linker searches
QMAKE_LFLAGS_DEBUG   += /VERBOSE:LIB
QMAKE_LFLAGS_RELEASE += /VERBOSE:LIB

DEFINES += _CRT_SECURE_NO_WARNINGS

SOURCES += \
    main.cpp
