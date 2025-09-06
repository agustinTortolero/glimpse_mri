# vtk_dicom.pro — Qt + DCMTK + VTK (Qt glue/OpenGL/Volume) with DLL copy & debug prints
QT       += widgets gui opengl openglwidgets
CONFIG   += c++17 console
TEMPLATE  = app

SOURCES += \
    main.cpp \
    MainWindow.cpp \
    DicomLoader.cpp

HEADERS += \
    MainWindow.h \
    DicomLoader.h

# Put exe under debug/ or release/
CONFIG(debug, debug|release) { DESTDIR = debug } else { DESTDIR = release }

# ==========================================================
# Windows + MSVC + vcpkg (two roots allowed)
# ==========================================================
win32:msvc {

    # Allow overrides (Qt Creator Env → Build Steps → Additional arguments)
    isEmpty(VCPKG_VTK_ROOT)   { VCPKG_VTK_ROOT   = C:/src/vcpkg }
    isEmpty(VCPKG_DCMTK_ROOT) { VCPKG_DCMTK_ROOT = C:/AgustinTortolero_repos/MRI/vcpkg }

    message(=== vcpkg roots ===)
    message(VCPKG_VTK_ROOT   = $$VCPKG_VTK_ROOT)
    message(VCPKG_DCMTK_ROOT = $$VCPKG_DCMTK_ROOT)

    TRIPLET = x64-windows

    # ----------------
    # DCMTK locations
    # ----------------
    DCMTK_INC = $$VCPKG_DCMTK_ROOT/installed/$$TRIPLET/include
    DCMTK_LIB = $$VCPKG_DCMTK_ROOT/installed/$$TRIPLET/lib
    DCMTK_BIN = $$VCPKG_DCMTK_ROOT/installed/$$TRIPLET/bin

    message(--- DCMTK ---)
    message(DCMTK_INC = $$DCMTK_INC)
    message(DCMTK_LIB = $$DCMTK_LIB)
    message(DCMTK_BIN = $$DCMTK_BIN)

    INCLUDEPATH += $$DCMTK_INC

    # --- DCMTK libraries (add JPEG/JPEG-LS to fix unresolveds) ---
    LIBS        += -L$$DCMTK_LIB \
                   ofstd.lib oflog.lib \
                   dcmdata.lib dcmimgle.lib dcmimage.lib \
                   dcmjpeg.lib dcmjpls.lib dcmtkcharls.lib ijg8.lib ijg12.lib ijg16.lib \
                   ws2_32.lib netapi32.lib

    # Avoid <windows.h> min/max macros
    DEFINES += NOMINMAX

    # -----------
    # VTK wiring
    # -----------
    VTK_VER      = 9.3
    VTK_INC_BASE = $$VCPKG_VTK_ROOT/installed/$$TRIPLET/include
    VTK_LIB_REL  = $$VCPKG_VTK_ROOT/installed/$$TRIPLET/lib
    VTK_LIB_DBG  = $$VCPKG_VTK_ROOT/installed/$$TRIPLET/debug/lib
    VTK_BIN_REL  = $$VCPKG_VTK_ROOT/installed/$$TRIPLET/bin
    VTK_BIN_DBG  = $$VCPKG_VTK_ROOT/installed/$$TRIPLET/debug/bin

    message(--- VTK ---)
    message(VTK_VER   = $$VTK_VER)
    message(VTK_INC   = $$VTK_INC_BASE/vtk-$$VTK_VER)
    INCLUDEPATH += $$VTK_INC_BASE $$VTK_INC_BASE/vtk-$$VTK_VER

    # Choose libdir and detect 'd' suffix on Debug
    CONFIG(debug, debug|release) {
        VTK_LIBDIR = $$VTK_LIB_DBG
        VTK_DSUF   =
        exists($$VTK_LIBDIR/vtkCommonCore-$${VTK_VER}d.lib) {
            VTK_DSUF = d
            message("VTK debug libs use 'd' suffix")
        } else {
            message("VTK debug libs have NO 'd' suffix")
        }
    } else {
        VTK_LIBDIR = $$VTK_LIB_REL
        VTK_DSUF   =
    }

    # --- VTK set for DICOM volume rendering with Qt ---
    LIBS += \
        $${VTK_LIBDIR}/vtksys-$${VTK_VER}$${VTK_DSUF}.lib \
        $${VTK_LIBDIR}/vtkCommonCore-$${VTK_VER}$${VTK_DSUF}.lib \
        $${VTK_LIBDIR}/vtkCommonMath-$${VTK_VER}$${VTK_DSUF}.lib \
        $${VTK_LIBDIR}/vtkCommonTransforms-$${VTK_VER}$${VTK_DSUF}.lib \
        $${VTK_LIBDIR}/vtkCommonDataModel-$${VTK_VER}$${VTK_DSUF}.lib \
        $${VTK_LIBDIR}/vtkCommonExecutionModel-$${VTK_VER}$${VTK_DSUF}.lib \
        $${VTK_LIBDIR}/vtkIOCore-$${VTK_VER}$${VTK_DSUF}.lib \
        $${VTK_LIBDIR}/vtkIOImage-$${VTK_VER}$${VTK_DSUF}.lib \
        $${VTK_LIBDIR}/vtkImagingCore-$${VTK_VER}$${VTK_DSUF}.lib \
        $${VTK_LIBDIR}/vtkRenderingCore-$${VTK_VER}$${VTK_DSUF}.lib \
        $${VTK_LIBDIR}/vtkRenderingFreeType-$${VTK_VER}$${VTK_DSUF}.lib \
        $${VTK_LIBDIR}/vtkRenderingOpenGL2-$${VTK_VER}$${VTK_DSUF}.lib \
        $${VTK_LIBDIR}/vtkRenderingUI-$${VTK_VER}$${VTK_DSUF}.lib \
        $${VTK_LIBDIR}/vtkInteractionStyle-$${VTK_VER}$${VTK_DSUF}.lib \
        $${VTK_LIBDIR}/vtkGUISupportQt-$${VTK_VER}$${VTK_DSUF}.lib \
        $${VTK_LIBDIR}/vtkRenderingVolume-$${VTK_VER}$${VTK_DSUF}.lib \
        opengl32.lib

    # ----------------------
    # Post-link: copy DLLs
    # ----------------------
    EXE_DIR  = $$OUT_PWD/$$DESTDIR
    COPY_LOG = $$EXE_DIR/_post_copy.log

    CONFIG(debug, debug|release) {
        QMAKE_POST_LINK += $$quote(cmd /c "echo [POST] EXE_DIR=\"$$EXE_DIR\" > \"$$COPY_LOG\"")
        QMAKE_POST_LINK += $$quote(cmd /c "echo [POST] Copy DEBUG DCMTK DLLs from $$DCMTK_BIN >> \"$$COPY_LOG\"")
        QMAKE_POST_LINK += $$quote(cmd /c "robocopy \"$$DCMTK_BIN\" \"$$EXE_DIR\" *.dll /XO /XN /NP /NJH /NJS /LOG+:\"$$COPY_LOG\" & if errorlevel 8 exit 1")

        QMAKE_POST_LINK += $$quote(cmd /c "echo [POST] Copy DEBUG VTK DLLs from $$VTK_BIN_DBG >> \"$$COPY_LOG\"")
        QMAKE_POST_LINK += $$quote(cmd /c "robocopy \"$$VTK_BIN_DBG\" \"$$EXE_DIR\" *.dll /XO /XN /NP /NJH /NJS /LOG+:\"$$COPY_LOG\" & if errorlevel 8 exit 1")

        QMAKE_POST_LINK += $$quote(cmd /c "echo [POST] dir /b of copied DLLs:>>\"$$COPY_LOG\" & dir /b \"$$EXE_DIR\\*.dll\" >> \"$$COPY_LOG\"")
    } else {
        QMAKE_POST_LINK += $$quote(cmd /c "echo [POST] EXE_DIR=\"$$EXE_DIR\" > \"$$COPY_LOG\"")
        QMAKE_POST_LINK += $$quote(cmd /c "echo [POST] Copy RELEASE DCMTK DLLs from $$DCMTK_BIN >> \"$$COPY_LOG\"")
        QMAKE_POST_LINK += $$quote(cmd /c "robocopy \"$$DCMTK_BIN\" \"$$EXE_DIR\" *.dll /XO /XN /NP /NJH /NJS /LOG+:\"$$COPY_LOG\" & if errorlevel 8 exit 1")

        QMAKE_POST_LINK += $$quote(cmd /c "echo [POST] Copy RELEASE VTK DLLs from $$VTK_BIN_REL >> \"$$COPY_LOG\"")
        QMAKE_POST_LINK += $$quote(cmd /c "robocopy \"$$VTK_BIN_REL\" \"$$EXE_DIR\" *.dll /XO /XN /NP /NJH /NJS /LOG+:\"$$COPY_LOG\" & if errorlevel 8 exit 1")

        QMAKE_POST_LINK += $$quote(cmd /c "echo [POST] dir /b of copied DLLs:>>\"$$COPY_LOG\" & dir /b \"$$EXE_DIR\\*.dll\" >> \"$$COPY_LOG\"")
    }
}

# Extra logging in Debug
CONFIG(debug, debug|release) {
    DEFINES += QT_MESSAGELOGCONTEXT
}

