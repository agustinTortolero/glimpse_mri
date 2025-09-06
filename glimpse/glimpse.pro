QT       += widgets
CONFIG   += c++17 console
TEMPLATE  = app

SOURCES += \
    main.cpp \
    MainWindow.cpp \
    DicomLoader.cpp

HEADERS += \
    MainWindow.h \
    DicomLoader.h

# --- Put exe in "debug/" or "release/" explicitly ---
CONFIG(debug, debug|release) {
    DESTDIR = debug
} else {
    DESTDIR = release
}

# =========================
# vcpkg + DCMTK (Windows)
# =========================
win32 {
    # Your local vcpkg root (override with env VCPKG_ROOT if desired)
    isEmpty(VCPKG_ROOT) {
        VCPKG_ROOT = C:/AgustinTortolero_repos/MRI/vcpkg
    }

    # -------------------------
    # MSVC branch (Qt MSVC kit)
    # -------------------------
    win32:msvc {
        TRIPLET    = x64-windows
        DCMTK_INC  = $$VCPKG_ROOT/installed/$$TRIPLET/include
        DCMTK_LIB  = $$VCPKG_ROOT/installed/$$TRIPLET/lib
        DCMTK_BIN  = $$VCPKG_ROOT/installed/$$TRIPLET/bin

        message(Using MSVC + vcpkg triplet $$TRIPLET)
        message(DCMTK_INC is $$DCMTK_INC)
        message(DCMTK_LIB is $$DCMTK_LIB)
        message(DCMTK_BIN is $$DCMTK_BIN)

        INCLUDEPATH += $$DCMTK_INC
        LIBS        += -L$$DCMTK_LIB \
                       dcmdata.lib dcmimgle.lib dcmimage.lib ofstd.lib oflog.lib \
                       ws2_32.lib netapi32.lib

        # Copy DLLs from vcpkg bin to our exe dir (dynamic triplet)
        EXE_DIR = $$OUT_PWD/$$DESTDIR
        COPY_LOG = $$EXE_DIR/_copy_dcmtk.log

        QMAKE_POST_LINK += $$quote(cmd /c "echo [POST] EXE_DIR=\"$$EXE_DIR\" DCMTK_BIN=\"$$DCMTK_BIN\" > \"$$COPY_LOG\"")
        QMAKE_POST_LINK += $$quote(cmd /c "robocopy \"$$DCMTK_BIN\" \"$$EXE_DIR\" *.dll /XO /XN /NP /NJH /NJS /LOG+:\"$$COPY_LOG\" & if errorlevel 8 exit 1")
        QMAKE_POST_LINK += $$quote(cmd /c "echo [POST] dir /b of DLLs:>>\"$$COPY_LOG\" & dir /b \"$$EXE_DIR\\*.dll\" >> \"$$COPY_LOG\"")

        # Avoid <windows.h> min/max macros clashing with std::min/max
        DEFINES += NOMINMAX
    }

    # -----------------------------
    # MinGW branch (kept as backup)
    # -----------------------------
    win32:!msvc {
        TRIPLET    = x64-mingw-dynamic
        DCMTK_INC  = $$VCPKG_ROOT/installed/$$TRIPLET/include
        DCMTK_LIB  = $$VCPKG_ROOT/installed/$$TRIPLET/lib
        DCMTK_BIN  = $$VCPKG_ROOT/installed/$$TRIPLET/bin

        message(Using MinGW + vcpkg triplet $$TRIPLET)
        message(DCMTK_BIN is $$DCMTK_BIN)

        INCLUDEPATH += $$DCMTK_INC
        LIBS        += -L$$DCMTK_LIB \
                       -ldcmdata -ldcmimgle -ldcmimage -lofstd -loflog \
                       -lws2_32 -lnetapi32

        EXE_DIR = $$OUT_PWD/$$DESTDIR
        COPY_LOG = $$EXE_DIR/_copy_dcmtk.log

        QMAKE_POST_LINK += $$quote(cmd /c "echo [POST] EXE_DIR=\"$$EXE_DIR\" DCMTK_BIN=\"$$DCMTK_BIN\" > \"$$COPY_LOG\"")
        QMAKE_POST_LINK += $$quote(cmd /c "robocopy \"$$DCMTK_BIN\" \"$$EXE_DIR\" *.dll /XO /XN /NP /NJH /NJS /LOG+:\"$$COPY_LOG\" & if errorlevel 8 exit 1")
        QMAKE_POST_LINK += $$quote(cmd /c "echo [POST] dir /b of DLLs:>>\"$$COPY_LOG\" & dir /b \"$$EXE_DIR\\*.dll\" >> \"$$COPY_LOG\"")
    }
}

CONFIG(debug, debug|release) {
    DEFINES += QT_MESSAGELOGCONTEXT
}
